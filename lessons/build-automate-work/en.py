# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-automate-work. Route: course/route/lessen/bouwen.yaml. Checks, the event logs and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-automate-work", "en")
    C = L.checks   # the made-up event logs of a webshop's returns, and the directly-follows graph
    F = C.facts()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [AI at work: centaur or cyborg](/course/open/build-ai-at-work?lang=en), and one about [One number per group: groupby](/course/open/data-groupby?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_frontier():
    quick_frontier = L.choice(
        "quick-frontier",
        question="A model got a task right in all eight tries. Another task, which looks just as easy, it got wrong every time. What does that show?",
        options={
            "luck": "That eight tries are too few to say anything",
            "jagged": "That the frontier is jagged: you find out per task, by trying",
            "broken": "That the model is broken, and you should not use it at all",
        },
        correct="jagged",
        why={
            "jagged": "Two tasks that look just as hard can lie on different sides of the frontier. So you try each task on cases you know.",
            "luck": "Eight out of eight against none out of eight is not luck. What does it say about the two tasks? See [AI at work: centaur or cyborg](/course/open/build-ai-at-work?lang=en).",
            "broken": "The same model did the first task right every time. What is the border between the two tasks called? See [AI at work: centaur or cyborg](/course/open/build-ai-at-work?lang=en).",
        },
    )
    quick_frontier
    return (quick_frontier,)


@app.cell(hide_code=True)
def quick_check_groupby():
    quick_groupby = L.choice(
        "quick-groupby",
        question='A table of shop sales has the columns `shop` and `amount`. What does `df.groupby("shop")["amount"].sum()` give?',
        options={
            "total": "One total: all amounts added up",
            "rows": "How many sales each shop had",
            "per": "The total amount of each shop, one number per shop",
        },
        correct="per",
        why={
            "per": "`groupby(\"shop\")` splits the rows by shop, `[\"amount\"]` takes their amounts, and `.sum()` gives one total per shop.",
            "total": "That is `df[\"amount\"].sum()`, without groups. What does `groupby` add? See [One number per group: groupby](/course/open/data-groupby?lang=en).",
            "rows": "Counting rows is `.size()` or `.count()`. What does `.sum()` do with the amounts of each shop? See [One number per group: groupby](/course/open/data-groupby?lang=en).",
        },
    )
    quick_groupby
    return (quick_groupby,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Much office work is the same steps, again and again: copy the details of an email into a system, send a label, check a form, pay. Software can take those steps over, and with AI it can even read the email. But software does exactly what you told it, also when the work does not go the way you thought. So before you automate, you look at how the work really goes.

    **An everyday example.** A school plans a new path across the grass, from the gate to the door. A few weeks before, the caretaker looks at where the grass is trodden flat: the children do not walk from the gate to the door, but from the bike shed. That is only a comparison; below you see how to find the trodden paths of work in a log.
    """)
    return


@app.cell(hide_code=True)
def predict_as_drawn():
    guess_as_drawn = L.guess(
        "as-drawn",
        question=f"A webshop drew its returns as five steps: return requested, label sent, package received, package checked, refund paid. Its system keeps a log of every step. Of the last {F['returns']} returns, how many went through exactly those five steps?",
        options={
            "all": "Nearly all of them",
            "half": "A little over half",
            "few": "Hardly any",
        },
        reveal=f"**A little over half: {F['as_drawn']} of the {F['returns']}.** In {F['no_label']} returns the customer sent the package without the label; in {F['photo_cases']} the check asked the customer for photos and checked again, sometimes more than once; {F['refused']} were refused; and {F['twice']} returns got their refund paid twice. None of that is in the drawing. The log is made up by the course, but logs of real companies show the same: work goes more ways than anyone drew.",
    )
    guess_as_drawn
    return (guess_as_drawn,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **no-code** | building an automation by clicking blocks together instead of writing code, such as *when an email with an invoice comes in, save the attachment and add a row to a sheet* |
    | **robotic process automation (RPA)** | software that repeats fixed clicks and keystrokes in programs you already have, as a person would; it follows rules and learns nothing |
    | **personal agent** | an assistant that takes steps for you, such as reading your mail and putting an appointment in your calendar, and chooses those steps itself |
    | **event log** | a table a system keeps of the work: one row for each step, with the case it belongs to, what happened (the activity) and when |
    | **process mining** | working out from an event log how a process really goes: which steps, in which order, and where the time goes |
    | **directly-follows graph** | a picture of a process from its event log: a box for each activity, and an arrow from one to another for every time it came right after it in a case |

    ### Three ways to automate

    | Way | What it does | Good for |
    |---|---|---|
    | **no-code** flow | a trigger and fixed steps, built from blocks in tools such as Zapier, Make, n8n or Power Automate; many now have a step that asks a language model | work that always goes the same way, between programs that have a connection to the tool |
    | **RPA** | a software robot that clicks through the screens of a program, often an old one without a connection; with machine learning added to read forms or sort emails, it is called *intelligent process automation* | fixed work in programs that nothing else can reach |
    | **personal agent** | an assistant that decides its own steps, with access to your mail, calendar or files; a later lesson shows how an agent works | work that differs every time, where you check what it did |

    The first two do exactly what you built, every time. The agent can handle the unexpected, and can also do something unexpected itself.

    ### Process mining

    Every system that handles cases, orders or returns writes down what happened: an event log. Process mining reads it. Its first picture is the directly-follows graph: for every two steps of one case that come right after each other, it counts the pair. A thick arrow is a way many cases go; a thin arrow that is not in the drawing is a detour; an arrow back is work done again. Process mining grew up largely at the Eindhoven University of Technology. *Task mining* does the same for the clicks of one person at a computer.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: a directly-follows graph

    1. **Put the log in order:** by case, and within a case by time. (The course's log already is.)
    2. **Go through the rows, two at a time:** row `i` and row `i + 1`.
    3. **Same case?** Then the two steps form a pair: `(steps[i], steps[i + 1])`. A new case starts? Then there is no pair: the last step of one return does not lead to the first of the next.
    4. **Count each pair,** and draw an arrow for it, as thick as its count.
    5. **Compare with the drawing:** which arrows are extra, which go back, which should never be there? And where does the time go? Only then decide what to automate.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A log of three returns, already in order:

    | case | activity | day |
    |---|---|---|
    | R001 | Return requested | 0 |
    | R001 | Label sent | 1 |
    | R001 | Package received | 4 |
    | R002 | Return requested | 0 |
    | R002 | Package received | 3 |
    | R003 | Return requested | 0 |
    | R003 | Label sent | 2 |
    | R003 | Package received | 5 |

    1. **In order:** yes, by case, then by day.
    2. **Two at a time:** rows 0 and 1, rows 1 and 2, and so on, up to rows 6 and 7: seven times.
    3. **Same case?** Rows 0 and 1: both R001, so the pair *(Return requested, Label sent)*. Rows 2 and 3: R001 and R002, a new case: no pair. In all, five pairs:
       *(Return requested, Label sent)*, *(Label sent, Package received)*, *(Return requested, Package received)*, *(Return requested, Label sent)*, *(Label sent, Package received)*.
    4. **Count:** *Return requested → Label sent* 2 times, *Label sent → Package received* 2 times, *Return requested → Package received* once.
    5. **Compare:** R002 went from the request straight to the package: the customer did not wait for the label. If a robot only ever waits for *Label sent*, it gets stuck on returns like R002.
    """)
    return


@app.cell(hide_code=True)
def text_which_automation():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which way to automate?
    """)
    return


@app.cell(hide_code=True)
def choice_which_automation_cell():
    choice_which_automation = L.choice(
        "which-automation",
        question="Every morning, someone at an insurer copies the details of new claims from an old program, which has no connection to anything, into a spreadsheet: the same twelve fields, in the same order. Which way of automating fits best?",
        options={
            "agent": "A personal agent, which decides by itself what to do with each claim",
            "rpa": "RPA: a software robot that clicks through the old program and copies the fields",
            "nocode": "A no-code flow that the old program starts when a claim comes in",
        },
        correct="rpa",
        why={
            "rpa": "Fixed steps in a program nothing else can reach: that is what RPA is for. The robot does exactly the same every morning, which is what you want here.",
            "agent": "An agent chooses its own steps, so it can also choose wrong ones. Do these steps ever change? Look at the table of the three ways.",
            "nocode": "A no-code flow needs a connection to the program, and this one has none. Look at the table of the three ways.",
        },
    )
    choice_which_automation
    return (choice_which_automation,)


@app.cell(hide_code=True)
def text_exercise_pairs():
    mo.vstack([
        mo.md(rf"""
        ### Exercise 2. The pairs of the log

        The box reads the webshop's log of {F['returns']} returns as `log`, with the columns `case`, `activity` and `day`, in order by case and day: {C.number(F['rows'], 0, L.lang)} rows. `cases` and `steps` are its columns as lists. Finish the loop: for every two rows that follow each other and belong to the same return, add the pair `(steps[i], steps[i + 1])` to `pairs`. Replace the `...`. The last line is `pairs`, so the check sees it; under the box, the course draws your graph. The check also tries a log of 40 other returns.
        """),
        L.task("Make `pairs`: every two steps of one return that directly follow each other."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pairs():
    box_pairs = L.code_box("pairs", lines=9)
    box_pairs
    return (box_pairs,)


@app.cell(hide_code=True)
def check_pairs(box_pairs):
    result_pairs = L.check("pairs", box_pairs.value)
    result_pairs
    return (result_pairs,)


@app.cell(hide_code=True)
def hints_pairs():
    L.hints(
        "pairs",
        [
            "Steps 2 and 3 of the recipe: a loop over `i`, and a test whether rows `i` and `i + 1` belong to the same return.",
            "`range(len(steps) - 1)` lets `i` go up to the last but one row, so that `i + 1` is always a row. The test compares `cases[i]` with `cases[i + 1]`.",
            "`for i in range(___):` with inside it `if ___ == ___:` and inside that `pairs.append((___, ___))`.",
        ],
        why=f"`for i in range(len(steps) - 1):` goes through the rows two at a time; `if cases[i] == cases[i + 1]:` keeps only two rows of the same return; `pairs.append((steps[i], steps[i + 1]))` adds their steps as one pair. The {C.number(F['rows'], 0, L.lang)} rows of {F['returns']} returns give {C.number(F['pairs'], 0, L.lang)} pairs: for each return, one fewer than its rows.",
    )
    return


@app.cell(hide_code=True)
def text_graph_after():
    mo.vstack([
        mo.Html(C.course_graph(L.lang)),
        mo.md(rf"""
        This is the graph of all {F['returns']} returns. The thickest arrows are the drawn way. But {F['photos']} times a check was followed by *Photos asked*, and each time by a check again: work done twice, in {F['photo_cases']} returns. {F['no_label']} returns went from the request straight to the package. And a few arrows are thin, but not harmless.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_took():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Where the time goes

        Each return starts on day 0, so the day of its last step is how many days it took. Make `took`: for each return, the largest `day`. Replace the `...`. The last line is `took`, so the check sees it. Under the box, the course splits your days into the returns where photos were asked and the rest.
        """),
        L.task("Make `took`: the largest day of each return."),
    ])
    return


@app.cell(hide_code=True)
def exercise_took():
    box_took = L.code_box("took", lines=6)
    box_took
    return (box_took,)


@app.cell(hide_code=True)
def check_took(box_took):
    result_took = L.check("took", box_took.value)
    result_took
    return (result_took,)


@app.cell(hide_code=True)
def hints_took():
    L.hints(
        "took",
        [
            "One number per return: that is `groupby`, as in [One number per group: groupby](/course/open/data-groupby?lang=en).",
            "Group by `case`, take the column `day`, and ask for its largest value, not its mean.",
            "`took = log.groupby(___)[___].max()`.",
        ],
        why=f"`log.groupby(\"case\")` splits the rows by return, `[\"day\"]` takes their days, and `.max()` gives the last one: how long each return took. The returns with photos asked took {C.number(F['days_photos'], 1, L.lang)} days on average, the others {C.number(F['days_other'], 1, L.lang)}.",
    )
    return


@app.cell(hide_code=True)
def text_took_after():
    mo.md(rf"""
    The returns where photos were asked took {C.number(F['days_photos'], 1, L.lang)} days on average; the others {C.number(F['days_other'], 1, L.lang)}. That loop is where the time goes. A robot that copies the details of each return faster would not touch it; asking for the photos with the return itself, before the package is sent, might.
    """)
    return


@app.cell(hide_code=True)
def text_never_twice():
    mo.md(r"""
    ### Exercise 4. An arrow that should not be there
    """)
    return


@app.cell(hide_code=True)
def choice_never_twice_cell():
    choice_never_twice = L.choice(
        "never-twice",
        question="Which arrow in the graph of exercise 2 shows a mistake that costs the webshop money?",
        options={
            "photos": "Package checked → Photos asked",
            "twice": "Refund paid → Refund paid",
            "nolabel": "Return requested → Package received",
        },
        correct="twice",
        why={
            "twice": f"A refund paid right after a refund: {F['twice']} customers got their money back twice. A thin arrow, so easy to miss, and nobody drew it. Before you automate the payments, you find out why.",
            "photos": "Asking for photos costs time, not money paid out by mistake. Which arrow pays something more than once?",
            "nolabel": "The customer sent the package without the label: harmless, the package still came in. Which arrow pays something more than once?",
        },
    )
    choice_never_twice
    return (choice_never_twice,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [One number per group: groupby](/course/open/data-groupby?lang=en) you summarized a table per group; here every return is a group, and the order of its rows matters too. In [AI at work: centaur or cyborg](/course/open/build-ai-at-work?lang=en) you split your own work into tasks; process mining does that for the work of a whole company, from its log. Later in the domain *Building with AI*, you build an agent yourself, and see why it needs checks that a fixed flow does not.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_never_twice, choice_which_automation, result_pairs, result_took):
    L.wrap_up(choice_which_automation, result_pairs, result_took, choice_never_twice)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
