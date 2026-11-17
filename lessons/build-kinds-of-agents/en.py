# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-kinds-of-agents. Route: course/route/lessen/bouwen.yaml. Checks, the course's runs, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-kinds-of-agents", "en")
    C = L.checks   # the four worlds and their tools, the course's runs of Qwen3, the checks and the demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [The agent loop](/course/open/build-agent-loop?lang=en), and one about [Searching, deep research and learning with AI](/course/open/build-research-and-learn?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_stop():
    quick_stop = L.choice(
        "quick-stop",
        question="When does `run_agent` stop asking the model?",
        options={
            "done": "When the model writes that it is done, or when every tool has been called once",
            "stop": "When an answer has no tool calls, or when it has sent `max_steps` requests",
            "error": "When a tool gives back an error, since the model cannot go on after that",
        },
        correct="stop",
        why={
            "stop": "An answer without tool calls is the model's answer to the task; the step limit stops a model that keeps calling tools.",
            "done": "The loop does not read what the model writes; it looks only at whether the answer has tool calls. See [The agent loop](/course/open/build-agent-loop?lang=en).",
            "error": "An error goes back to the model as the tool's result, and the model can try something else. See [The agent loop](/course/open/build-agent-loop?lang=en).",
        },
    )
    quick_stop
    return (quick_stop,)


@app.cell(hide_code=True)
def quick_check_source():
    quick_source = L.choice(
        "quick-source",
        question="A deep-research report gives a page as the source of a claim. When is that source worth something?",
        options={
            "known": "When the page is on a well-known site, so that the claim on it is probably right",
            "twice": "When the report gives the same page for more claims, since it must have read it",
            "opened": "Once you have opened the page and found the claim there",
        },
        correct="opened",
        why={
            "opened": "A model can name a page it never read, or a page that says something else. Only the page itself shows it.",
            "known": "A well-known site can say something other than the report claims, or nothing about it. See [Searching, deep research and learning with AI](/course/open/build-research-and-learn?lang=en).",
            "twice": "A model can name the same page again and again without having read it. See [Searching, deep research and learning with AI](/course/open/build-research-and-learn?lang=en).",
        },
    )
    quick_source
    return (quick_source,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A coding agent, a deep-research agent, a personal agent in your mail and a shopping agent look like four different things. Under them is the same loop you wrote in [The agent loop](/course/open/build-agent-loop?lang=en): send the messages and the tools, run the call in the answer, add its result, and again. What makes them different is which tools they get, and what may go wrong with those tools. A coding agent can break a project, a research agent can name a source it never read, a personal agent can send a mail in your name, and a shopping agent can spend your money. So each kind needs its own limit, in the code that runs its tools.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **agentic commerce** | shopping and paying done by agents on behalf of a person, within limits the person has set |
    | **mandate** | what a person allows an agent beforehand, such as in which shops it may pay by itself and how much per order |
    | **test report** | what a test run gives back: how many tests passed, and for each failure what the call gave and what was expected |
    | **`except Exception as e:`** | catches an error of any kind, and gives it the name `e`; `str(e)` is its message |
    | **`x not in items`** | `True` when `x` is not in the list `items`; `x in items` is its opposite |
    | **`a and b`** | `True` only when `a` and `b` are both `True` |

    The course built four small worlds, each with its own tools. A coding agent works in a project with one file, `shop.py`, and tests that check it. A research agent reads the pages of a town's website, `town.example`. A personal agent has a mailbox and a calendar. A shopping agent searches two web shops, `shop.example` and `deals.example`, fills a cart and pays. All four run with the same loop, the same model and the same step limit. Only the system prompt, the tools and the limit differ:
    """)
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.idea_picture("en"))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### What the course measured

    The course gave Qwen3-0.6B three tasks in each world, at temperature 0, so that the same request always gets the same answer, with at most one tool call per answer and at most {s} requests per task. In the first request of each task, `tool_choice` was `"required"`, so that the agent starts with a tool call; the later requests left the choice to the model. The coding agent's tasks are three bugs in `shop.py`. The research agent's are three questions about the town; one of the pages is an old forum post that gives another opening time. The personal agent puts an appointment from a mail in the calendar, answers a colleague by mail, and says what is in the calendar. The shopping agent gets a mandate in its system prompt: it may pay by itself only at `shop.example`, up to {m} per order.
    """.format(s=F["steps"], m=C.euro(C.MANDATE["max"], "en")))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture("en"))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.vstack([
        mo.md(C.measured_text("en")),
        L.watch_out("A limit in the system prompt is a request to the model; the limit in the code that runs the tool is the one that holds. Give each kind of agent only the tools its tasks need, and put the step that cannot be undone, such as sending or paying, behind the user's yes or a mandate the user set."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: a new kind of agent

    1. **Start from the loop you have:** the same requests, the same step limit.
    2. **Give it the tools of its world,** and no more: what it must read, and what it must change.
    3. **Ask what can go wrong with each tool,** and what cannot be undone: a deleted file, a sent mail, a payment.
    4. **Put the limit in the code that runs the tool:** the tests stay as they are, a source counts only once it is opened, a mail waits for the user's yes, a payment stays within the mandate.
    5. **Give the agent results it can act on:** a test report that says what the call gave and what was expected, an error that says what is allowed.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(C.worked_text("en"))
    return


@app.cell(hide_code=True)
def picture_worked():
    mo.Html(C.worked_picture(*C.worked_run(), "en"))
    return


@app.cell(hide_code=True)
def text_report():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The report a coding agent reads

        `test_report(name, fn, cases)` runs the tests of a function and writes the report that goes back to the coding agent as the result of `run_tests`. `name` is the name of the function, `fn` the function itself, and `cases` a list of pairs: an argument, and what the function should give for it. The first lines go through the cases, write each call as text, such as `total([110, 240])`, and put the report together at the end. For each case, call `fn` with the argument. When it gives something other than expected, add a line such as `total([110, 240]) gave 351, expected 350.` to `failed`. When it stops with an error, add a line such as `cheapest([]) stopped with an error: list index out of range.`, where the part after the colon is `str(e)`. The check tries the function of task 1 before and after the fix, and the functions of tasks 2 and 3.
        """),
        L.task("Replace `...` with a `try` block that calls `fn` and adds a line when the value is not the expected one, and an `except Exception as e:` block that adds a line for the error."),
    ])
    return


@app.cell(hide_code=True)
def exercise_report():
    box_report = L.code_box("report", lines=12)
    box_report
    return (box_report,)


@app.cell(hide_code=True)
def check_report(box_report):
    result_report = L.check("report", box_report.value)
    result_report
    return (result_report,)


@app.cell(hide_code=True)
def hints_report():
    L.hints(
        "report",
        [
            "Two things can happen for each case: `fn` gives back a value, which can be wrong, or it stops with an error. `try` and `except` handle the second; inside the `try`, compare the value.",
            "In the `try`, keep what `fn(argument)` gives in `got`, and compare it with `expected`. The line for `failed` joins `call`, `\" gave \"`, `str(got)`, `\", expected \"`, `str(expected)` and `\".\"` with `+`.",
            "In the `try`, once `got` holds the value: `if ___:`, with the `failed.append(...)` under it.",
        ],
        why="""```python
def test_report(name, fn, cases):
    failed = []
    for argument, expected in cases:
        call = name + "(" + str(argument) + ")"
        try:
            got = fn(argument)
            if got != expected:
                failed.append(call + " gave " + str(got) + ", expected " + str(expected) + ".")
        except Exception as e:
            failed.append(call + " stopped with an error: " + str(e) + ".")
    passed = len(cases) - len(failed)
    return " ".join([str(passed) + " of " + str(len(cases)) + " tests passed."] + failed)
```

The agent fixes the code from what the report says: which call failed, what it gave, and what it should have given.""",
    )
    return


@app.cell(hide_code=True)
def text_unread():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Sources it never opened

        A research agent names its sources in square brackets, such as `[3]`. `unread(answer, opened)` gives the numbers of the sources the answer names that are not in `opened`, the list of pages the agent read: each number once, in the order of the answer. `answer.split("[")` cuts the text at each `[`; every part after the first starts with a number and `]`. The check tries the answer of the picture and four more.
        """),
        L.task("Replace `...` with a line that turns the part before `]` into a number, and an `if` that adds that number to `missing` when it is not in `opened` and not in `missing` yet."),
    ])
    return


@app.cell(hide_code=True)
def exercise_unread():
    box_unread = L.code_box("unread", lines=8)
    box_unread
    return (box_unread,)


@app.cell(hide_code=True)
def check_unread(box_unread):
    result_unread = L.check("unread", box_unread.value)
    result_unread
    return (result_unread,)


@app.cell(hide_code=True)
def hints_unread():
    L.hints(
        "unread",
        [
            "A part such as `\"5]. On Sunday …\"` starts with the number. Cut it at `]`, and keep what comes before.",
            "`part.split(\"]\")` gives a list of the pieces around `]`, and `int` makes a number of a text. Then test the number against both lists with `not in`, joined with `and`.",
            "First `number = int(___)`, then the `if` with `missing.append(number)` under it.",
        ],
        why="""```python
def unread(answer, opened):
    missing = []
    for part in answer.split("[")[1:]:
        number = int(part.split("]")[0])
        if number not in opened and number not in missing:
            missing.append(number)
    return missing
```

A source the agent never opened is a claim it did not check. Your code can catch that without reading the pages: it only compares the numbers.""",
    )
    return


@app.cell(hide_code=True)
def text_pay():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Paying within the mandate

        The mandate is a dictionary: `mandate["shops"]` is the list of shops where the agent may pay by itself, and `mandate["max"]` is the most an order may cost. `may_pay(cart, mandate)` gets the cart as a list of dictionaries, each with `"product"`, `"shop"`, `"price"` and `"quantity"`. It gives back `"no"` when a product comes from a shop that is not in the mandate, `"ask"` when the total is above the maximum, so that the user decides, and `"yes"` when the agent may pay by itself. The first lines add up the total. The check tries the carts of the picture and one more, with a mandate that allows both shops.
        """),
        L.task("Add a rule in the loop for a shop that is not in the mandate, and after the loop the rules that decide with the total."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pay():
    box_pay = L.code_box("may-pay", lines=11)
    box_pay
    return (box_pay,)


@app.cell(hide_code=True)
def check_pay(box_pay):
    result_pay = L.check("may-pay", box_pay.value)
    result_pay
    return (result_pay,)


@app.cell(hide_code=True)
def hints_pay():
    L.hints(
        "may-pay",
        [
            "A shop outside the mandate rules the whole cart out, whatever it costs, so that check comes first: inside the loop, for each item. The total is known only after the loop.",
            "In the loop, before the line with `total`: `return \"no\"` when the item's `\"shop\"` is not in `mandate[\"shops\"]`. After the loop: `\"ask\"` when `total` is above `mandate[\"max\"]`, else `\"yes\"`.",
            "In the loop, before the line with `total`: `if ___:`, with `return \"no\"` under it.",
        ],
        why="""```python
def may_pay(cart, mandate):
    total = 0
    for item in cart:
        if item["shop"] not in mandate["shops"]:
            return "no"
        total = total + item["price"] * item["quantity"]
    if total > mandate["max"]:
        return "ask"
    return "yes"
```

Within the mandate the agent pays by itself, so the user is not asked about every small order; above it, the user decides. Exactly the maximum is still allowed: `>` and not `>=`.""",
    )
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it on Qwen3

    Choose a task, and Qwen3-0.6B does it with the same loop, in that task's world: the course starts its own server on your computer, with the model behind it. Your functions are in place: the coding agent's tests report with your `test_report`, the shopping agent pays only when your `may_pay` says `"yes"`, and under the research agent's answer, your `unread` says which sources it never opened. A coding task needs exercise 1, a research task exercise 2, and a shopping task exercise 3.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1.5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*The chat model Qwen3-0.6B is not on your computer yet. The button downloads it once from Hugging Face, `Qwen/Qwen3-0.6B` at a fixed version, 1.5 GB, under the Apache-2.0 license, with the tokenizer of Qwen3-0.6B-Base.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 1.5 GB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    mo.md("*Qwen3-0.6B is downloaded.*")
    return


@app.cell(hide_code=True)
def task_cell():
    _kinds = {"coding": "Coding", "research": "Research", "personal": "Personal", "shopping": "Shopping"}
    task_pick = mo.ui.dropdown(options={f"{_kinds[k]}: {t}": (k, i) for k in C.KINDS for i, t in enumerate(C.TASKS[k])},
                               value=f"Coding: {C.TASKS['coding'][0]}", label="Task")
    run_button = mo.ui.run_button(label="Start the agent")
    mo.vstack([task_pick, run_button])
    return run_button, task_pick


@app.cell(hide_code=True)
def demo_run(box_pay, box_report, box_unread, run_button, task_pick):
    _kind, _task = task_pick.value
    _report = box_report.value if C.learner_fn("report", box_report.value) is not None else None
    _unread = C.learner_fn("unread", box_unread.value)
    _pay = C.learner_fn("may-pay", box_pay.value)
    _needs = {"coding": (_report, 1), "research": (_unread, 2), "shopping": (_pay, 3)}
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _kind in _needs and _needs[_kind][0] is None:
        _out = mo.md("*Finish exercise {k} first: this task uses your own function.*".format(k=_needs[_kind][1]))
    elif not run_button.value:
        _out = mo.md("*Press the button to start the agent.*")
    else:
        mo.output.replace(mo.md("*The agent has started; the server loads Qwen3 …*"))
        _names = {"coding": "coding agent", "research": "research agent", "personal": "personal agent", "shopping": "shopping agent"}
        _words = {"title": "Qwen3 as a {k}, with the same loop".format(k=_names[_kind]), "step": "step {k}: the model's call",
                  "no_call": "the model's answer, without a call", "limit": "The agent reached the step limit of {s} requests.",
                  "unread": "Your unread: the answer names {m}, which the agent never opened.",
                  "all_read": "Your unread: the agent opened every source the answer names.",
                  "stopped": "Your function stopped with {e}.", "broke": "The run stopped with {e}.",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.run_agent(_kind, _task, _words, report_code=_report, unread_fn=_unread, pay_fn=_pay))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The loop is the one from [The agent loop](/course/open/build-agent-loop?lang=en); only the tools, the system prompt and the limit changed. The coding agent is the one you worked with in [Working with a coding agent: plan, spec and context](/course/open/build-coding-workflow?lang=en), the research agent does what [Searching, deep research and learning with AI](/course/open/build-research-and-learn?lang=en) described, and the personal agent is the one from [Automating work: flows, personal agents and process mining](/course/open/build-automate-work?lang=en), which chooses its own steps, so you check what it did. The next lesson looks back at the agents before these, and why most of them went astray.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_pay, result_report, result_unread):
    L.wrap_up(result_report, result_unread, result_pay)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
