# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-error-analysis. Route: course/route/lessen/bouwen.yaml. Checks, the recorded replies, the course's notes and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-error-analysis", "en")
    C = L.checks   # the bank's assistant, its recorded replies, the course's notes on them, the checks and the pictures
    NOTES = C.NOTES
    COUNTS = C.COUNTS


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [JSON you can rely on](/course/open/build-structured-output?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_json():
    quick_json = L.choice(
        "quick-json",
        question="In JSON mode, all eight of Qwen3's answers about a bike were valid JSON. What did that tell you?",
        options={
            "fits": "That every answer had the right keys and types",
            "form": "That json.loads could read them, nothing more",
            "right": "That the details in the answers were right",
        },
        correct="form",
        why={
            "form": "Valid JSON is not yet the JSON your app needs: in four of those answers the price was still a text. And whether what an answer says is right, no check of its form can tell you.",
            "fits": "JSON mode only keeps the text JSON: in four of the eight answers the price was still a text. Keys and types are what a JSON schema checks. See [JSON you can rely on](/course/open/build-structured-output?lang=en).",
            "right": "JSON mode looks at the form, not at the meaning: valid JSON can hold a wrong price. See [JSON you can rely on](/course/open/build-structured-output?lang=en).",
        },
    )
    quick_json
    return (quick_json,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    An assistant answers the customers of an online bank in the chat of its app. You try it on 120 real questions, and every answer is valid JSON with a topic and a reply. Is the assistant good? A number tells you only what you thought of measuring. What goes wrong in this app, such as a reply that blames the bank or a step the app does not have, you find only by reading the answers.

    **An everyday example.** A teacher grades a stack of essays. Not with a list of possible mistakes made beforehand, but by reading each essay and writing a word in the margin. Afterwards the margins show that half the class mixes up the same two words: that is the lesson to give next.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **error analysis** | reading the answers of an app one by one, and writing down in a few words what goes wrong in each |
    | **failure taxonomy** | the kinds of mistakes you found, each with a rule for what belongs to it, and how often each happened |
    | **evaluation-driven development** | building an app by measuring before and after every change, and keeping a change only when the measurements show that it helps; what you measure comes from the kinds you found |

    The kinds come from the answers, not from a list you make in advance: such a list names mistakes such as "invalid JSON", and misses those of your app. One answer can have more than one kind. Reading takes time, so you read a sample of about a hundred answers; the biggest kinds already show up in it.
    """)
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: error analysis

    1. **Collect:** about a hundred real inputs, and your app's answers to them.
    2. **Read and note:** for each answer, write down in a few words what goes wrong, or nothing when it is right.
    3. **Sort:** put notes that say the same thing into one kind, and give each kind a name and a rule.
    4. **Count:** for each kind, the number of answers that have it.
    5. **Choose:** fix the biggest kind first, and measure again after the change.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.vstack([
        mo.md(r"""
        ### Worked example

        1. **Collect:** 120 questions of real customers of an online bank, 20 for each of six topics, from the dataset BANKING77. The assistant is Qwen3 with this system prompt, at temperature 0, so that the same request always gets the same answer:
        """),
        mo.Html(C.prompt_html(C.SYSTEM)),
        mo.md(r"""
        2. **Read and note:** three of the replies, with what the course wrote down for each:
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_worked():
    mo.vstack([mo.Html(C.answer_card(_i, L.lang)) for _i in C.WORKED])
    return


@app.cell(hide_code=True)
def text_worked_sort():
    _f = C.findings()
    mo.md(r"""
    3. **Sort:** the note on answer 4 holds three kinds: the wrong topic, the steps of another problem, and a mistake of the bank that the customer never mentioned. Answer 80 says something the facts do not say: *made up*. Answer 2 has no note.
    4. **Count:** the course read all {n} replies this way: {noted} had at least one kind, {fine} had none.
    5. **Choose:** the biggest kind is the wrong topic, with {t} replies. Exercise 4 asks what to do about it.
    """.format(n=_f["n"], noted=_f["noted"], fine=_f["fine"], t=_f["counts"]["wrong-topic"]))
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    mo.md(r"""
    ### What the course measured

    All {n} replies were valid JSON with a topic and a reply, and all {n} ended by themselves, with `finish_reason` `"stop"`. Measured by its form, the assistant is done. Read, {noted} of the {n} replies had something wrong. The course sorted its notes into six kinds:
    """.format(n=_f["n"], noted=_f["noted"]))
    return


@app.cell(hide_code=True)
def picture_kinds():
    mo.Html(C.kinds_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_kinds():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        Together the kinds count more than {noted}, because {multi} replies had more than one. The wrong topic and another problem often come together: {both} replies have both. {rr} of the {r} questions about a refund, and {pr} of the {p} about a blocked PIN, got the topic `card_payment_wrong_exchange_rate`, and {asks} of those {rate} replies asked for the date and amount of a payment. For a question the facts do not answer, the system prompt says to tell the customer that a person from the team will reply: none of the {n} replies did, not even for *{police}*, which the facts do not answer.
        """.format(noted=_f["noted"], multi=_f["noted"] - _f["one_kind"], both=_f["topic_and_other"], rr=_f["refunds_rate"], r=_f["refunds"],
                   pr=_f["pins_rate"], p=_f["pins"], asks=_f["rate_asks"], rate=_f["refunds_rate"] + _f["pins_rate"], n=_f["n"], police=C.QUESTIONS[99])),
        L.watch_out("Two people who read the same replies do not always write the same notes. Write down the rule of each kind, as in the table, and read again with it when you are unsure: a rule makes the counting repeatable."),
    ])
    return


@app.cell(hide_code=True)
def text_count():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Count the kinds

        `NOTES` holds the course's notes: a list for each reply, with the kinds the course wrote down for it, and `[]` for a reply without a mistake. `NOTES[3]`, the note on answer 4, is `["wrong-topic", "other-problem", "admits"]`. `count_kinds(notes)` should give a dictionary with, for each kind, the number of replies that have it. The start code puts in 1 for every kind. The check also tries short lists of its own.
        """),
        L.task("When a kind is already in `counts`, add 1 to it; otherwise start it at 1."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count_kinds", lines=9)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count_kinds", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count_kinds",
        [
            "Each time a kind comes along in the loop, its count should go up by one. A kind that comes along for the first time is not in `counts` yet.",
            "`kind in counts` is `True` when the kind is already there. Use it in an `if`, with an `else` for a new kind.",
            "Replace the line `counts[kind] = 1` by `if kind in counts:` with `counts[kind] = ___` under it, then `else:` with `counts[kind] = 1` under it.",
        ],
        why="""```python
def count_kinds(notes):
    counts = {}
    for kinds in notes:
        for kind in kinds:
            if kind in counts:
                counts[kind] = counts[kind] + 1
            else:
                counts[kind] = 1
    return counts
```

This is the failure taxonomy as numbers. A reply with two kinds counts once for each, so the counts add up to more than the number of replies with a mistake.""",
    )
    return


@app.cell(hide_code=True)
def text_biggest():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The biggest kind

        `biggest(counts)` should give the kind with the highest count. `COUNTS` holds the course's counts, the outcome of exercise 1. When two kinds have the same count, the first one in `counts` should come out. The start code gives the last kind, whatever its count.
        """),
        L.task("Let a kind become `best` only when its count is higher than `most`, and then make `most` its count."),
    ])
    return


@app.cell(hide_code=True)
def exercise_biggest():
    box_biggest = L.code_box("biggest", lines=8)
    box_biggest
    return (box_biggest,)


@app.cell(hide_code=True)
def check_biggest(box_biggest):
    result_biggest = L.check("biggest", box_biggest.value)
    result_biggest
    return (result_biggest,)


@app.cell(hide_code=True)
def hints_biggest():
    L.hints(
        "biggest",
        [
            "Go through the kinds and remember the best one so far, with its count in `most`. A kind beats it only when its count is higher.",
            "Inside the loop: `if counts[kind] > most:`, and under it the two lines that make this kind the best so far.",
            "Under the `if`: `best = ___` and `most = ___`, both one step further in than the `if`.",
        ],
        why="""```python
def biggest(counts):
    best = None
    most = 0
    for kind in counts:
        if counts[kind] > most:
            best = kind
            most = counts[kind]
    return best
```

With `>`, a later kind with the same count does not take the place of the first. For the course's notes, the biggest kind is `"wrong-topic"`.""",
    )
    return


@app.cell(hide_code=True)
def text_which():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Which kind?

        One more reply, without the course's note:
        """),
        mo.Html(C.answer_card(C.WHICH, L.lang, note=False)),
    ])
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which-kind",
        question="Which kind does this reply belong to?",
        options={
            "elsewhere": "sends elsewhere",
            "made-up": "made up",
            "other-problem": "another problem",
        },
        correct="made-up",
        why={
            "made-up": "The facts say nothing about finding a card, and the app has no way to do it: the reply promises what the bank cannot do, and asks for a card number in a chat. The facts do not answer this question, so the reply should have said that a person from the team will reply.",
            "elsewhere": "It does not send the customer anywhere: it promises to look for the card itself. That is something the facts do not say: *made up*.",
            "other-problem": "It does not ask for what belongs to another problem, such as the date and amount of a payment. It promises what the facts do not say: *made up*.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_fix():
    _f = C.findings()
    mo.md(r"""
    ### Exercise 4. What first?

    The biggest kind is the wrong topic, with {t} replies; {both} of them also asked for what belongs to another problem.
    """.format(t=_f["counts"]["wrong-topic"], both=_f["topic_and_other"]))
    return


@app.cell(hide_code=True)
def choice_fix_cell():
    _f = C.findings()
    choice_fix = L.choice(
        "fix-first",
        question="Which change do you try first?",
        options={
            "admit": "Add a sentence: never say that a payment was wrong before the team checked it",
            "json": "Ask for JSON mode, so that every answer is valid JSON",
            "topics": "Put each fact next to its topic, so the model takes the facts of one topic",
        },
        correct="topics",
        why={
            "topics": "{rr} of the {r} questions about a refund and {pr} of the {p} about a blocked PIN got the topic of a wrong exchange rate, and their replies took its facts. In the system prompt, nothing ties a fact to its topic. Tying them aims at both kinds at once. Whether it helps, you measure: that is the next lesson.".format(
                rr=_f["refunds_rate"], r=_f["refunds"], pr=_f["pins_rate"], p=_f["pins"]),
            "admit": "That aims at the kind *admits a mistake*, with {a} replies: a real problem, but a smaller one than the wrong topic, with {t}. Fix the biggest kind first.".format(
                a=_f["counts"]["admits"], t=_f["counts"]["wrong-topic"]),
            "json": "All {n} replies were valid JSON already: JSON mode fixes a problem the assistant does not have. Without reading, you might have measured only that, and called the assistant done.".format(n=_f["n"]),
        },
    )
    choice_fix
    return (choice_fix,)


@app.cell(hide_code=True)
def text_read():
    mo.md(r"""
    ### Read the replies yourself

    Choose a group, and go through its replies one at a time, with the course's note. Read a few of the made-up ones: would you have put each in the same kind?
    """)
    return


@app.cell(hide_code=True)
def settings_group():
    group_pick = mo.ui.dropdown(options=C.group_options(L.lang), value=list(C.group_options(L.lang))[0], label="Group", full_width=True)
    group_pick
    return (group_pick,)


@app.cell(hide_code=True)
def settings_reply(group_pick):
    reply_pick = mo.ui.slider(start=1, stop=max(1, len(C.group(group_pick.value))), value=1, label="Reply", show_value=True)
    reply_pick
    return (reply_pick,)


@app.cell(hide_code=True)
def demo_read(group_pick, reply_pick):
    _numbers = C.group(group_pick.value)
    mo.Html(C.answer_card(_numbers[min(reply_pick.value, len(_numbers)) - 1], L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The kinds you found are what to measure from now on. [An eval set and regression tests](/course/open/build-eval-set?lang=en) turns the kinds that code can see, such as the wrong topic, into checks in code, and measures whether a fix for the biggest kind helps. A kind such as *made up* needs someone who reads the meaning: [Who grades the answer](/course/open/build-judges?lang=en) lets a model do that, and measures how often it agrees with the course's notes.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fix, choice_which, result_biggest, result_count):
    L.wrap_up(result_count, result_biggest, choice_which, choice_fix)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
