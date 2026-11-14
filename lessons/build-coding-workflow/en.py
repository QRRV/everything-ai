# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-coding-workflow. Route: course/route/lessen/bouwen.yaml. Checks, the 32 programs and their picture: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-coding-workflow", "en")
    C = L.checks   # the 32 programs Qwen3 wrote from four ways of asking, what they give, and the picture


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [From autocomplete to agents: help with code](/course/open/build-coding-help?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kind():
    quick_kind = L.choice(
        "quick-kind",
        question="You want a new option in your program, spread over three files, with a test, while you do something else. Which kind of help fits?",
        options={
            "agent": "A coding agent",
            "completion": "Code completion",
            "chat": "A chat about your code",
        },
        correct="agent",
        why={
            "agent": "Work across files, with tests to run, without you doing each step: that is what a coding agent is for. This lesson is about how you give it such a task.",
            "completion": "Code completion suggests a line where you type; the files and the test stay your work. See [From autocomplete to agents: help with code](/course/open/build-coding-help?lang=en).",
            "chat": "A chat answers, and you do the work yourself. See [From autocomplete to agents: help with code](/course/open/build-coding-help?lang=en).",
        },
    )
    quick_kind
    return (quick_kind,)


@app.cell(hide_code=True)
def quick_check_ask():
    quick_ask = L.choice(
        "quick-ask",
        question="Which of these should a coding agent ask you about before it does it?",
        options={
            "tests": "Running the tests",
            "install": "Installing a package from the internet",
            "search": "Searching the project's code",
        },
        correct="install",
        why={
            "install": "Code from strangers would run on your computer, and that is hard to undo. Running the tests and searching change nothing you cannot undo.",
            "tests": "Running the tests changes nothing you cannot undo, and the agent needs it to check its own work. See [From autocomplete to agents: help with code](/course/open/build-coding-help?lang=en).",
            "search": "Searching changes nothing at all. See [From autocomplete to agents: help with code](/course/open/build-coding-help?lang=en).",
        },
    )
    quick_ask
    return (quick_ask,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A coding agent does what you ask, and fills in what you did not ask with guesses. In a small task you see the guesses at once; in a large task they hide in files you never open. How you hand over the task decides how much it has to guess, and how soon you find out where it guessed wrong.

    **An everyday example.** You have a builder redo your kitchen. *Make it nicer* gets you a kitchen, but not the one you meant. A drawing with measurements gets you closer. Going through the plan together before a wall comes down catches a missing socket while that still costs nothing. And the notes about the house, where the water pipes run, save the builder from guessing. That is only a comparison; below you see what Qwen3 did with four ways of asking for the same small program.
    """)
    return


@app.cell(hide_code=True)
def predict_examples():
    guess_examples = L.guess(
        "spec-examples",
        question="The course gave Qwen3 a precise description with three examples of what must come out, and had it write the program eight times. How many of the eight programs got all three of those examples right?",
        options={
            "half": "About half",
            "none": "None",
            "all": "All eight",
        },
        reveal="**None.** Seven of the eight got the first example right and all eight the third, but not one the second: *'+31 6 1234 5678' gives '0612345678'*. The example was right there in the description. Below you see what would have caught it at once.",
    )
    guess_examples
    return (guess_examples,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **spec** | short for specification: what a program must do, written down before the code, with examples, the edges too |
    | **spec-driven development** | working with a coding agent from a spec: first write down what must come out, then let the agent build and test against it |
    | **plan** | the steps an agent intends to take, written down before it changes anything, so that you can read and correct them |
    | **context** | everything a model has in front of it when it answers: your request, the files it read, and notes such as AGENTS.md |

    ### Four parts of a task

    | Part | What it says | Why it helps |
    |---|---|---|
    | the spec | *what* must come out, with examples | an example cannot be read in two ways, and you can try it |
    | the plan | *how*, in steps, in an order | you read it before there is code, while fixing it costs nothing |
    | small steps | one change at a time, with the tests after each | a mistake shows up in the step that made it |
    | the context | what holds for every task, such as how to run the tests | the agent starts each task with it, and you do not repeat it |

    Many agents read a file named AGENTS.md before your request; Claude Code reads CLAUDE.md, which can point to AGENTS.md. Examples in November 2026: Kiro and GitHub's Spec Kit build the whole way of working around a spec, with a list of tasks that are checked off one by one; Claude Code and Cursor have a plan mode, in which the agent only plans and changes nothing until you agree.

    **The run.** The course asked Qwen3-0.6B in four ways for the same small function, eight times each: one that cleans up a Dutch phone number.

    - **Vague request:** *Write a Python function clean_phone(text) that cleans up a phone number.*
    - **Spec with examples:** *Write a Python function clean_phone(text) for Dutch phone numbers. Keep only the digits; a number that starts with +31 or 0031 starts with 0 instead; return the 10 digits as a string, or an empty string when there are not 10 digits left. Examples: '06-12345678' gives '0612345678', '+31 6 1234 5678' gives '0612345678', '12345' gives ''.*
    - **Spec, plan first:** the same spec, ending with *First write a short plan in numbered steps, then the code.*
    - **Project notes, vague request:** the vague request, with the same rules and examples before it as notes about the project, the way an agent reads AGENTS.md before your request.

    A *string* is a piece of text; an empty string, '', is text with nothing in it. The course tried every program on nine numbers, three of them the spec's own examples, without changing a line of it:
    """)
    return


@app.cell(hide_code=True)
def picture_ways():
    mo.vstack([
        mo.Html(C.ways_grid(L.lang)),
        mo.md(C.grid_after(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def picture_code():
    L.explore("program-code", "program", lambda _v: C.program_frame(int(_v), L.lang), start=20, low=1, high=len(C.OUTPUTS), step=1, role="",
              task="Slide through the 32 programs: eight from the vague request, then eight from the spec, eight with a plan first, and eight with project notes. For the programs with a plan, read the plan too: does it have a step for +31?")
    return


@app.cell(hide_code=True)
def text_plans():
    mo.md(C.plans_after(L.lang))
    return


@app.cell(hide_code=True)
def text_moves():
    L.watch_out("A large coding agent writes this function right far more often than Qwen3-0.6B. But a large task leaves much more to guess, and an agent can also say that the tests pass without having run them. So you look at what the tests say yourself, just as here every program was run instead of believed.")
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: a task for a coding agent

    1. **Let it look first.** Ask it to read the files that matter and tell you what it found, without changing anything.
    2. **Write the spec:** what must come out, with examples, the edges included, such as too short, too long and the odd cases.
    3. **Ask for a plan, and read it** before any code is written. Is a step missing, or in the wrong order? Say so now.
    4. **Go in small steps.** After each step the tests run, the spec's examples among them, and you look at what changed.
    5. **Put what always holds in AGENTS.md,** such as how to run the tests and what it must not touch, so that the next task starts with it.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    You have a coding agent tidy up the phone numbers in your club's member list: some are written with dashes, some with +31.

    1. **Look first:** *Read the project and tell me where phone numbers are stored and used. Change nothing yet.* It finds them in two files: the member list, and the page that shows it.
    2. **Spec:** you write the rules from the run above, with five examples: the three from the spec, plus '0031612345678' gives '0612345678' and '06 1234 567' gives '', because it is too short.
    3. **Plan:** its plan has three steps: remove everything that is not a digit, check for 10 digits, save. You see that +31 is missing: after the first step, '+31 6 1234 5678' becomes '31612345678', 11 digits, and is thrown away. You ask for a step before it: first turn +31 or 0031 into 0.
    4. **Small steps:** the agent turns your five examples into tests and runs them after each step. You read what the tests say yourself, not only what the agent says about them.
    5. **AGENTS.md:** you add one line, *Phone numbers are stored as 10 digits, starting with 0.* The next task that touches phone numbers starts with that.
    """)
    return


@app.cell(hide_code=True)
def text_spec_plan_context():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Spec, plan or AGENTS.md?

    You hand an agent the task of cleaning up the phone numbers of your club. Where does each sentence belong?
    """)
    return


@app.cell(hide_code=True)
def exercise_spec_plan_context():
    sort_parts = L.sort("spec-plan-context")
    sort_parts
    return (sort_parts,)


@app.cell(hide_code=True)
def check_spec_plan_context(sort_parts):
    result_parts = L.check("spec-plan-context", sort_parts)
    result_parts
    return (result_parts,)


@app.cell(hide_code=True)
def hints_parts():
    L.hints(
        "spec-plan-context",
        [
            "Ask of each sentence: does it say what must come out, how to get there, or what holds in this project whatever the task?",
            "A spec is about the result of this one task, a plan is steps in an order, and AGENTS.md holds what is true for every task.",
            "Two sentences in each group. *An example* and *Too short* say what must come out; *First* and *Then* are steps.",
        ],
        why="Spec: **An example** and **Too short**. Plan: **First** and **Then**. AGENTS.md: **Tests** and **The data folder**.",
    )
    return


@app.cell(hide_code=True)
def text_which_helped():
    mo.md(r"""
    ### Exercise 2. What helped most?
    """)
    return


@app.cell(hide_code=True)
def choice_which_helped_cell():
    choice_which_helped = L.choice(
        "which-helped",
        question="Which change to the vague request helped Qwen3 the most?",
        options={
            "notes": "The project notes before the request",
            "spec": "The spec with examples",
            "plan": "The plan first, on top of the spec",
        },
        correct="spec",
        why={
            "spec": "From 11 to 39 of the 72 numbers. What a model has to guess, it often guesses wrong, and a spec with examples leaves much less to guess. The plan on top of it added nothing here: 38.",
            "notes": "The notes did help: 11 became 21. Compare that with the totals under the other columns.",
            "plan": "Look at the totals at the bottom of the picture: *Spec, plan first* against *Spec with examples*. What did the plan add?",
        },
    )
    choice_which_helped
    return (choice_which_helped,)


@app.cell(hide_code=True)
def text_plus_31():
    mo.md(r"""
    ### Exercise 3. What would have caught it?
    """)
    return


@app.cell(hide_code=True)
def choice_plus_31_cell():
    choice_plus_31 = L.choice(
        "plus-31",
        question="Not one of the 32 programs turned +31 into 0, not even the 24 that had the spec, with its example. What would have caught that at once?",
        options={
            "tests": "The spec's examples as tests, run after every step",
            "longer": "A longer spec, with the rule about +31 written twice",
            "ask": "Asking the agent whether its code is right",
        },
        correct="tests",
        why={
            "tests": "A test made of the example *'+31 6 1234 5678' gives '0612345678'* fails on all 32 programs the first time it runs. Run after every step, the tests also catch a step that breaks what already worked.",
            "longer": "The rule was already in the spec, with an example, and all 24 programs that had it missed it. Saying it twice is still only saying it: what would *show* you that it went wrong?",
            "ask": "In ten answers, the example already stands under the code, as if the code gives it. None of those programs does. What a model says about its code is not a check.",
        },
    )
    choice_plus_31
    return (choice_plus_31,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [From autocomplete to agents: help with code](/course/open/build-coding-help?lang=en) you chose the kind of help and set an agent up; here you gave it a task: a spec to check against, a plan to read, small steps, and AGENTS.md for what always holds. Your examples only help once they run: later in this module, [Checking code that AI wrote](/course/open/build-check-ai-code?lang=en) shows how to write them as tests.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_plus_31, choice_which_helped, result_parts):
    L.wrap_up(result_parts, choice_which_helped, choice_plus_31)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
