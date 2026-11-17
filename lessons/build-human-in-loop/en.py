# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-human-in-loop. Route: course/route/lessen/bouwen.yaml. Checks, the bank's tools, the day of messages, the person and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-human-in-loop", "en")
    C = L.checks   # the bank's tools, the day of messages with their steps, the person and the pictures
    TOOLS = C.TOOLS     # per tool: "money" (costs money), "sends" (sends something out), "can_undo"
    DAY = C.DAY         # ten messages of one day, each with the "steps" the course wrote out for it
    STEPS = C.STEPS     # all steps of the day in a row; a step is {"id", "tool", "arguments"}
    PERSON = C.PERSON   # the person at the bank: PERSON(step) gives "yes", "no" or the step as they changed it


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [What a tool may do](/course/open/build-tool-permissions?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_ask():
    quick_ask = L.choice(
        "quick-ask",
        question="A policy says `ask` for `delete_file`, and the agent calls `delete_file`. What happens?",
        options={
            "after": "It runs, and the user is told after it",
            "never": "It is refused, whatever the user says",
            "yes": "It runs only after the user says yes",
        },
        correct="yes",
        why={
            "yes": "`ask` means: ask the user first, and run the call only on yes. `allow` runs at once, and `deny` never runs.",
            "after": "Then it would already be too late to say no. `ask` asks before the call runs, and runs it only on yes. See [What a tool may do](/course/open/build-tool-permissions?lang=en).",
            "never": "That is `deny`. With `ask` the user decides, and the call runs on yes. See [What a tool may do](/course/open/build-tool-permissions?lang=en).",
        },
    )
    quick_ask
    return (quick_ask,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    mo.md(r"""
    ## Why this matters

    In [What a tool may do](/course/open/build-tool-permissions?lang=en), a policy said per tool whether it may run, must ask first, or may not run. Now the bank's assistant acts for its customers: it looks up an account, blocks a card, orders a new one, pays money back, sends an email, closes an account. Which of those steps should wait for a person?

    Ask too seldom, and a refund of 250 euros goes out where the customer asked for 25. Ask at every step, and on a day of {tasks} messages someone at the bank is asked {every} times, also before reading a balance. Every ask costs that person time, and the customer a wait.

    **An everyday example.** A new employee in a shop may answer questions, look things up and take back an item with its receipt on their own. A refund without a receipt goes past the manager first. The manager is not called for every customer: then there would be no time left for the ones that matter.
    """.format(tasks=_f["tasks"], every=_f["asks"]["every"]))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **human-in-the-loop** | a setup in which a person takes part in what an AI system does: here, someone who approves, changes or refuses a step of the agent before it runs |
    | **approval step** | a point where the agent stops before a step and waits for a person's answer: yes, a changed step, or no |

    Before each step, the code asks three questions about its tool: does it cost money, does it send something out of the bank, can it not be undone? If none is true, the step runs at once: a mistake there costs little and can be put right. If one is true, the agent stops and asks a person. The person can say yes, change the step, such as its amount, or say no. On no, the agent stops and tells the customer that a person will reply, because the steps after the refused one may depend on it.

    The questions are about the tool, not about the model, so they sit in the code, like the policy of [What a tool may do](/course/open/build-tool-permissions?lang=en). The person's answers are worth keeping: every change and every no is a step the agent planned wrong, a case for the eval set of [An eval set and regression tests](/course/open/build-eval-set?lang=en).
    """)
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.flow_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: an approval step

    1. **Per tool:** write down what it does to the world: whether it costs money, sends something, and can be undone.
    2. **In the code that runs the steps:** ask before every step that costs money, sends something or cannot be undone; run the rest at once.
    3. **Show the whole step:** the tool and its arguments, such as the amount and the address, as it will run.
    4. **Three answers:** yes runs the step, a changed step runs as changed, no stops the agent, which tells the customer that a person will reply.
    5. **Keep the answers:** count how often a person changed or refused a step, and put those steps in your eval set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _f = C.findings()
    _task = DAY[2]
    mo.md(r"""
    ### Worked example

    `TOOLS` describes the {k} tools of the bank's agent: for each, whether it costs money, sends something and can be undone. `TOOLS["order_card"]` is `{order}`: a new card costs the bank money, but the order can be canceled.

    `DAY` holds {tasks} messages of one day, each with the steps the course wrote out for it, as an agent might plan them; `STEPS` has all {steps} in a row. `DAY[2]` is *{message}*, and its steps are `{s1}` and `{s2}` with an amount of {amount}: a 0 too many.

    `PERSON` stands for someone at the bank. `PERSON(step)` gives `"yes"`, `"no"`, or the step as they changed it. For this refund it gives the same step with an amount of {fixed}.
    """.format(k=len(TOOLS), order="{" + ", ".join(f'"{_k}": {_v}' for _k, _v in TOOLS["order_card"].items()) + "}", tasks=_f["tasks"], steps=_f["steps"], message=_task["message"],
               s1=_task["steps"][0]["tool"], s2=_task["steps"][1]["tool"], amount=L.number(_task["steps"][1]["arguments"]["amount"], 2),
               fixed=L.number(PERSON(_task["steps"][1])["arguments"]["amount"], 2)))
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    _a = _f["answers"]["rule"]
    mo.md(r"""
    ### What the rule does on the day

    The course wrote {wrong} of the {steps} steps wrong on purpose, and wrote what the person answers: the refund of 250 euros, closing an account without calling the customer first, blocking a card that the customer found again, and sending receipts to an address that came in the message.

    Asked by the rule, the person was asked {rule} times: {yes} yes, {changed} changed and {no} no. Three wrong steps were caught: the refund ran as the person changed it, and the other two did not run. The fourth, blocking the card, ran without anyone being asked, because a block can be undone: `unblock_card` puts it right. Asked at every step, the person catches that one too, at {every} asks instead of {rule}. Never asked, all {wrong} ran.
    """.format(wrong=_f["wrong"], steps=_f["steps"], rule=_f["asks"]["rule"], yes=_a["yes"], changed=_a["changed"], no=_a["no"], every=_f["asks"]["every"]))
    return


@app.cell(hide_code=True)
def picture_measured():
    mo.vstack([
        mo.Html(C.ways_picture(L.lang)),
        L.watch_out("A changed step runs as the person left it, so show the person the whole step, with its amount and its address, not only the name of the tool."),
    ])
    return


@app.cell(hide_code=True)
def text_needs():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. When to ask

        `needs_approval(step, tools)` should say whether the agent must ask before `step`: `True` when its tool costs money, sends something, or cannot be undone. `tools[step["tool"]]` is the tool's dictionary, with `"money"`, `"sends"` and `"can_undo"`, each `True` or `False`. `a or b` is `True` when at least one of the two is `True`, and `not x` is `True` when `x` is `False`. The start code never asks. The picture runs your function on every tool.
        """),
        L.task("Return `True` when one of the three is true: money, sends, or cannot be undone."),
    ])
    return


@app.cell(hide_code=True)
def exercise_needs():
    box_needs = L.code_box("needs_approval", lines=4)
    box_needs
    return (box_needs,)


@app.cell(hide_code=True)
def check_needs(box_needs):
    result_needs = L.check("needs_approval", box_needs.value)
    result_needs
    return (result_needs,)


@app.cell(hide_code=True)
def hints_needs():
    L.hints(
        "needs_approval",
        [
            "Three questions, and one yes is enough to ask.",
            "Join the three with `or`; the third question asks whether a step can *not* be undone.",
            "Make it `return tool[\"money\"] or tool[\"sends\"] or ___`.",
        ],
        why="""```python
def needs_approval(step, tools):
    tool = tools[step["tool"]]
    return tool["money"] or tool["sends"] or not tool["can_undo"]
```

A look-up and a block run at once: a mistake there can be put right. A new card asks because it costs the bank money, even though the order can be canceled.""",
    )
    return


@app.cell(hide_code=True)
def text_run():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Ask, and go on as the person says

        `run_steps(steps, ask)` should run the steps of one message as the person answers. `ask(step)` asks the person and gives `"yes"`, `"no"`, or the step as they changed it. Ask only before a step for which `needs_approval(step, TOOLS)` is `True`, and count how often you ask. On `"no"`, stop: the steps after it do not run. On a changed step, run that one instead. Give back the steps that ran and the number of asks. The start code runs every step and never asks. The picture runs your function on every message of the day, with `PERSON`.
        """),
        L.task("Before a step that needs approval, add 1 to `asks` and ask; stop on no, and run a changed step as changed."),
    ])
    return


@app.cell(hide_code=True)
def exercise_run():
    box_run = L.code_box("run_steps", lines=13)
    box_run
    return (box_run,)


@app.cell(hide_code=True)
def check_run(box_run):
    result_run = L.check("run_steps", box_run.value)
    result_run
    return (result_run,)


@app.cell(hide_code=True)
def hints_run():
    L.hints(
        "run_steps",
        [
            "Three answers, three outcomes: yes runs the step, a changed step runs as changed, and no stops the agent.",
            "Inside an `if needs_approval(step, TOOLS):`, add 1 to `asks` and ask with `answer = ask(step)`; what follows depends on `answer`.",
            "On no, end the loop at once: `if answer == \"no\":` with `___` under it.",
        ],
        why="""```python
def run_steps(steps, ask):
    done = []
    asks = 0
    for step in steps:
        if needs_approval(step, TOOLS):
            asks = asks + 1
            answer = ask(step)
            if answer == "no":
                break
            if answer != "yes":
                step = answer
        done.append(step)
    return done, asks
```

On no the agent stops instead of going on: the email that says the account is closed must not go out when the account is still open.""",
    )
    return


@app.cell(hide_code=True)
def text_which():
    mo.md(r"""
    ### Exercise 3. Which step waits?
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which-step",
        question="Which of these steps of the bank's agent should wait for a person?",
        options={
            "unblock": "Unblock the card again",
            "refund": "Pay back 40 euros",
            "read": "Read the last ten transactions",
        },
        correct="refund",
        why={
            "refund": "Paying back costs money and cannot be undone: that is what a person approves. Reading changes nothing, and an unblock can be undone with `block_card`.",
            "unblock": "An unblock can be undone: `block_card` blocks the card again. Paying back 40 euros costs money and cannot be undone; that is the step to ask about.",
            "read": "Reading changes nothing in the world. Paying back 40 euros costs money and cannot be undone; that is the step to ask about.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_every():
    mo.md(r"""
    ### Exercise 4. Ask at every step?
    """)
    return


@app.cell(hide_code=True)
def choice_every_cell():
    _f = C.findings()
    _every, _rule = _f["asks"]["every"], _f["asks"]["rule"]
    choice_every = L.choice(
        "every-step",
        question="To be safe, the bank lets its agent ask a person before every step, also before a look-up. What is the risk?",
        options={
            "habit": "The person gets used to saying yes",
            "slow": "The agent can no longer use its tools",
            "none": "None: more asks always catch more",
        },
        correct="habit",
        why={
            "habit": "On the course's day that is {e} asks instead of {r}, and {x} of them are about steps that cost nothing and can be undone. Someone who approves that many harmless steps can get used to saying yes, also to the step that matters. Ask where it counts.".format(
                e=_every, r=_rule, x=_every - _rule),
            "slow": "The tools still run, each after a yes. The risk is the person: {e} asks a day instead of {r}, most of them about harmless steps, and someone who approves that many can get used to saying yes.".format(
                e=_every, r=_rule),
            "none": "On the course's day, asking at every step caught one more wrong step, a block that could be undone, for {x} more asks. Someone who approves that many harmless steps can get used to saying yes, also to the step that matters.".format(
                x=_every - _rule),
        },
    )
    choice_every
    return (choice_every,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The rule is a policy as in [What a tool may do](/course/open/build-tool-permissions?lang=en), now based on what a step does to the world. Every change and every no is a case for the eval set of [An eval set and regression tests](/course/open/build-eval-set?lang=en). [How agents go wrong](/course/open/build-agent-failures?lang=en) counts the ways an agent goes wrong in recorded runs; an approval step is one of the things that help against them.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_every, choice_which, result_needs, result_run):
    L.wrap_up(result_needs, result_run, choice_which, choice_every)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
