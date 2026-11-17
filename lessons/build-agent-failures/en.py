# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-agent-failures. Route: course/route/lessen/bouwen.yaml. Checks, the twenty runs, what the course found in each and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-agent-failures", "en")
    C = L.checks   # the twenty recorded runs, the ways the course found in each, and the pictures
    RUNS = C.RUNS  # each run: "name", "task", "calls", "results", "answer", "steps", "limit"; a shop run also "wanted" and "added"


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Look at the mistakes first](/course/open/build-error-analysis?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kinds():
    quick_kinds = L.choice(
        "quick-kinds",
        question="In error analysis, where do the kinds of mistakes come from?",
        options={
            "list": "From a list made before reading",
            "model": "From what the model says went wrong",
            "notes": "From notes on the answers",
        },
        correct="notes",
        why={
            "notes": "You read about a hundred answers, write down in a few words what goes wrong in each, and sort those notes into kinds. The kinds come from your own answers.",
            "list": "A list made in advance finds what you expected and misses the rest. The kinds come from notes on the answers themselves. See [Look at the mistakes first](/course/open/build-error-analysis?lang=en).",
            "model": "The model does not know where it went wrong, and what it says about it is text like any other. The kinds come from your notes on the answers. See [Look at the mistakes first](/course/open/build-error-analysis?lang=en).",
        },
    )
    quick_kinds
    return (quick_kinds,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    mo.md(r"""
    ## Why this matters

    An agent takes many steps before it answers, and one wrong step can spoil the rest while the answer still sounds sure. In [The agent loop](/course/open/build-agent-loop?lang=en), Qwen3-0.6B answered {right} of the {k} kitchen questions right. To fix that, a count of right and wrong is not enough: you need to know *how* each run went wrong. A run that goes round in circles needs a different fix from a run that makes up what is in the pantry.

    The ways agents go wrong come back from agent to agent. This lesson reads {n} recorded runs of the course's agent and counts five of those ways. {k} of them are the kitchen runs of the agent loop. The other {s} are in a web shop: the agent got a task such as *Put eggs and butter in the cart.* and a page with buttons, and clicked buttons by their name until it said it was done.

    **An everyday example.** A new colleague who runs errands can go wrong in ways you will recognize: phoning the same supplier twice, buying at the wrong shop, saying the delivery came when it did not, coming back halfway through the list, or buying things nobody asked for. Each has its own fix: a list to tick off, a receipt, a look in the bag at the door.
    """.format(right=sum(r.startswith("kitchen") for r in _f["right"]), k=_f["kitchen"], n=_f["runs"], s=_f["shop"]))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **agent failure mode** | a way of going wrong that comes back across agents and tasks, such as looping or making up a result |

    Five of them come back in almost every agent, and each shows in a run's trace: its calls, their results and its answer.

    | Way | What the run shows | What helps |
    |---|---|---|
    | **stops too early** | it answers before it has looked up what the answer needs | code that checks the needed calls came before it takes the answer, and otherwise sends the model back with what is missing |
    | **does more than asked** | it does something the task did not ask for | code that checks each action against the task before it runs, and an approval step for what cannot be undone |
    | **loops** | it makes a call it made before, or goes on until the step limit | a step limit, and code that refuses a call made before and says so |
    | **makes up a result** | its answer states a result no tool gave, such as what the pantry holds or what is in the cart | code that writes the numbers into the answer itself, or a check that reads the answer beside the results |
    | **uses a tool wrongly** | it calls a tool that does not fit, or gives it an argument the tool does not know | arguments with a fixed list of choices, and a tool that says what it does not know instead of answering anyway |

    A run that answers without a result it needs, such as a shopping list without a look in the pantry, stops too early. It makes up a result when it says what a tool would have said.

    Code can find four of the five when it knows what the task needs and what the tools know: a call made before, a product nobody asked for, a needed call that never came and an argument the tool does not know are all things code can compare. A made-up result is harder: it often puts true numbers in a sentence that is false. Finding it takes reading, by you or by a judge as in [Who grades the answer](/course/open/build-judges?lang=en).

    The five are not a list to tick off instead of reading. Your own error analysis, as in [Look at the mistakes first](/course/open/build-error-analysis?lang=en), says which of them your agent has, and may find others.
    """)
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.modes_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: find how your agent goes wrong

    1. **Keep each run:** the calls with their arguments, their results and the answer, as a trace.
    2. **Read about twenty runs** and write down for each how it went wrong; one run can go wrong in several ways.
    3. **Write rules in code** for what code can see: a call made before, a run that reached the step limit, an argument the tool does not know, an action the task did not ask for, a needed call that never came.
    4. **Count per way,** and fix the biggest with a check in code where you can; read the rest, or have a judge read it.
    5. **Measure again** on the same runs after every change.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _run = RUNS[3]
    mo.md(r"""
    ### Worked example

    `RUNS` holds the {n} runs. Each is a dictionary with its `"name"`, the `"agent"` (`"kitchen"` or `"shop"`), the `"task"`, its `"calls"` in order, each a pair of the tool's name and its arguments, the `"results"`, the `"answer"` (`None` when the step limit stopped the run), the number of `"steps"` and the `"limit"`. A kitchen run also has the `"dish"`; a shop run also has `"wanted"`, the products the task asked for, and `"added"`, what each click put in the cart.

    `RUNS[3]` is kitchen 4: *{task}* Its calls are `{calls}`, and each result was *0 grams*. It goes wrong in three ways at once. In its first step it asked for the tomato twice: a call made before, so it loops. It asked the pantry for `"tomato"`, which the pantry knows as `"tomatoes"`, and for `"soup"`, which is a dish: it uses a tool wrongly. And it answered without looking up the recipe: it stops too early. Its answer says the pantry has nothing for tomato soup, while it holds {tomatoes} grams of tomatoes. The pantry made it worse: it answers *0 grams* for anything it does not know.
    """.format(n=len(RUNS), task=_run["task"], calls=C.code_of(_run["calls"]), tomatoes=C.W.PANTRY["tomatoes"]))
    return


@app.cell(hide_code=True)
def picture_worked_example():
    mo.Html(C.trace_picture("kitchen 4", L.lang))
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    _c = _f["count"]
    _text = mo.md(r"""
    ### What the course counted

    The course read each of the {n} runs and wrote down by hand how it went wrong; the grid above shows the result. {early} runs stopped too early: all {k} in the kitchen, since none looked in the pantry for every ingredient, and one in the shop. {after} kitchen runs answered after the recipe alone, {none_called} called no tool at all and wrote that they still needed the recipe, and the other {rest} looked up only part of what they needed. {more} shop runs put something in the cart that the task did not ask for, {loop} runs looped, {made} made up a result, and {wrong} used a tool wrongly. {fine} runs went right in every way, all of them in the shop.

    Then the course wrote a rule in code for each way except making up, and ran the rules on the same runs. For those four ways, the rules found exactly what the reading found. For making up, the course tried a rule that looks for numbers in the answer that no tool and no task gave: it found none in the {made} made-up answers. Two kitchen runs wrote the grams of the recipe as what is in the pantry: true numbers in a false sentence. Shop 6 wrote that it had added the pine nuts, without a number at all.
    """.format(n=_f["runs"], early=_c["too-early"], k=_f["kitchen"], after=len(_f["after_recipe"]), none_called=len(_f["no_calls"]),
               rest=_f["kitchen"] - len(_f["after_recipe"]) - len(_f["no_calls"]),
               more=_c["more"], loop=_c["loop"], made=_c["made-up"], wrong=_c["wrong-tool"], fine=len(_f["none"])))
    mo.vstack([
        _text,
        L.watch_out("One run can go wrong in several ways at once: shop 6 clicked the pasta twice, added rice, never added the pine nuts and said it had. Write down every way, not only the first one you see."),
    ])
    return


@app.cell(hide_code=True)
def text_repeats():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A call made before

        `repeats(calls)` should give the calls of a run that were made before in the same run: each call that is the same as an earlier one, name and arguments both, in order. A call is a pair, such as `("pantry", {"item": "tomato"})`, and `==` compares two pairs on both parts. `RUNS[3]["calls"]` has `("pantry", {"item": "tomato"})` twice, so `repeats(RUNS[3]["calls"])` should give that call once. The start code gives an empty list. The picture runs your function on all twenty runs.
        """),
        L.task("Put each call that was made before in `again`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_repeats():
    box_repeats = L.code_box("repeats", lines=8)
    box_repeats
    return (box_repeats,)


@app.cell(hide_code=True)
def check_repeats(box_repeats):
    result_repeats = L.check("repeats", box_repeats.value)
    result_repeats
    return (result_repeats,)


@app.cell(hide_code=True)
def hints_repeats():
    L.hints(
        "repeats",
        [
            "Keep a list of the calls you have seen so far: a call that is already in it was made before.",
            "The order matters: ask before you add the call to `seen`; the other way round, every call is already in it.",
            "Above `seen.append(call)`: `if ___:`, with `again.append(call)` under it.",
        ],
        why="""```python
def repeats(calls):
    seen = []
    again = []
    for call in calls:
        if call in seen:
            again.append(call)
        seen.append(call)
    return again
```

A call made before is one sign of a loop. The other is a run that goes on until the step limit: shop 5 clicked six different products and never made a call again.""",
    )
    return


@app.cell(hide_code=True)
def text_compare():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. What the shop agent left out, and what it added

        `compare(added, wanted)` should compare what a run put in the cart with what the task asked. Give back two lists: `missing`, what the task asked that is not in `added`, in the order of `wanted`; and `extra`, what is in `added` that the task did not ask, in the order of `added`. `x not in a_list` is `True` when `x` is not in the list. The start code gives two empty lists. The picture runs your function on the eight shop runs.
        """),
        L.task("Fill `missing` from `wanted`, and `extra` from `added`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_compare():
    box_compare = L.code_box("compare", lines=10)
    box_compare
    return (box_compare,)


@app.cell(hide_code=True)
def check_compare(box_compare):
    result_compare = L.check("compare", box_compare.value)
    result_compare
    return (result_compare,)


@app.cell(hide_code=True)
def hints_compare():
    L.hints(
        "compare",
        [
            "Two questions: what did the task ask that the cart does not have, and what does the cart have that the task did not ask?",
            "One loop over `wanted` for `missing`, and one loop over `added` for `extra`.",
            "The first loop: `for product in wanted:`, then `if ___:`, then `missing.append(product)`; the second the other way round.",
        ],
        why="""```python
def compare(added, wanted):
    missing = []
    extra = []
    for product in wanted:
        if product not in added:
            missing.append(product)
    for product in added:
        if product not in wanted:
            extra.append(product)
    return missing, extra
```

Something missing means the run stopped too early; something extra means it did more than asked. Shop 6 has both, and its answer said it had added the pine nuts: compare the cart with the task, not with what the model says.""",
    )
    return


@app.cell(hide_code=True)
def text_which():
    mo.md(r"""
    ### Exercise 3. Which way?
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which-mode",
        question="A kitchen run looked up the recipe of an omelette, never called `pantry`, and answered: *{said}* Which way of going wrong is that sentence?".format(
            said=C.pantry_sentence("kitchen 5")),
        options={
            "loop": "It loops",
            "made": "It makes up a result",
            "more": "It does more than it was asked",
        },
        correct="made",
        why={
            "made": "No tool said what is in the pantry: the numbers are the grams of the recipe, written as if they were in the pantry. A rule that looks for numbers no tool gave misses it, because the recipe gave them; finding it takes reading.",
            "loop": "It made one call, to the recipe, and answered. Nothing came twice, and the step limit was far away. The sentence says what is in the pantry, which no tool told it: it makes up a result.",
            "more": "It did less, not more: it never looked in the pantry. The sentence says what is in the pantry without any tool having said so: it makes up a result.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_helps():
    mo.md(r"""
    ### Exercise 4. What helps?
    """)
    return


@app.cell(hide_code=True)
def choice_helps_cell():
    choice_helps = L.choice(
        "what-helps",
        question="Shop 2 was asked for eggs and butter, and also put milk and cheese in the cart. What helps best against that?",
        options={
            "check": "Code checks each click against the task",
            "prompt": "A line in the prompt: add only what is asked",
            "limit": "A lower step limit",
        },
        correct="check",
        why={
            "check": "The task names the products, so code can refuse a click that adds anything else, every time. Where a task is not that clear, ask the user before the action, as in [Ask before acting](/course/open/build-human-in-loop?lang=en).",
            "prompt": "The system prompt already told the agent to stop when the cart holds what was asked, and shop 2 went on. A line in a prompt is a request to the model; a check in code holds every time: refuse a click that adds a product the task did not name.",
            "limit": "Shop 2 added the milk with its very first click, so a lower limit stops nothing that matters. A check in code before each click can refuse a product the task did not name.",
        },
    )
    choice_helps
    return (choice_helps,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The ways come from reading runs, as the kinds of [Look at the mistakes first](/course/open/build-error-analysis?lang=en) came from reading answers; the kitchen runs are those of [The agent loop](/course/open/build-agent-loop?lang=en). Where code cannot decide whether an action is wanted, the approval step of [Ask before acting](/course/open/build-human-in-loop?lang=en) helps against doing more than asked. [Prompt, RAG or fine-tune?](/course/open/build-prompt-rag-finetune?lang=en) asks which change to make when the model itself falls short.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_helps, choice_which, result_compare, result_repeats):
    L.wrap_up(result_repeats, result_compare, choice_which, choice_helps)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
