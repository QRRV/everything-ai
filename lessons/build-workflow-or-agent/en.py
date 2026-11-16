# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-workflow-or-agent. Route: course/route/lessen/bouwen.yaml. Checks, the course's runs, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-workflow-or-agent", "en")
    C = L.checks   # the kitchen, the course's runs of Qwen3 as an agent and as a workflow, the checks and the demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Tools as JSON: function calling](/course/open/build-tool-calling?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_results():
    quick_results = L.choice(
        "quick-results",
        question="Qwen3's answer holds a tool call of `convert`. What does your program put in the next request, after the messages so far?",
        options={
            "user": "Only the result, written as a new message of the user, so that the request stays short and cheap",
            "result": "The answer itself, and a message with the role `tool` holding the result and the call's `id`",
            "nothing": "Nothing new: the course's server keeps the call, runs `convert` itself and sends the result later",
        },
        correct="result",
        why={
            "result": "The model reads which call each result belongs to by its id, and goes on from there.",
            "user": "A result goes back with the role `tool` and the id of its call, right after the answer that made the call. See [Tools as JSON](/course/open/build-tool-calling?lang=en).",
            "nothing": "Your program runs `convert` itself, and sends back the result. See [Tools as JSON](/course/open/build-tool-calling?lang=en).",
        },
    )
    quick_results
    return (quick_results,)


@app.cell(hide_code=True)
def quick_check_auto():
    quick_auto = L.choice(
        "quick-auto",
        question="What does `\"tool_choice\": \"auto\"` leave to the model?",
        options={
            "tools": "Which tools go into the request, out of all the tools your program has",
            "schema": "Which arguments the function `convert` accepts, and of which types",
            "decide": "Whether it calls a tool at all, and if so, which one and with what",
        },
        correct="decide",
        why={
            "decide": "With `\"auto\"`, the model decides; with `convert` named, your program does.",
            "tools": "Your program puts the tools into the request; `\"auto\"` is about what the model does with them. See [Tools as JSON](/course/open/build-tool-calling?lang=en).",
            "schema": "The schema in the tool's description says that, and your program writes it. See [Tools as JSON](/course/open/build-tool-calling?lang=en).",
        },
    )
    quick_auto
    return (quick_auto,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [What an agent is](/course/open/agent-idea?lang=en), Qwen3 got two tools and a loop for a simple question: what to buy for pancakes. {why_runs} The model chose the steps, and chose them badly. Yet the steps of that task are always the same: look up the recipe, look in the pantry for each ingredient, and work out what is missing. Your code can take those steps itself, the same way every time, and leave to the model only what needs language. That is a workflow. Before you build an agent, ask whether you need one.
    """.format(why_runs=C.agent_idea_line("en")))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **workflow** | a program in which your code fixes the steps and their order, and the model fills in only some of them, such as reading the question or writing the answer; also called an *LLM workflow* |
    | **`text.lower()`** | the text in small letters: `"Pancakes".lower()` is `"pancakes"` |

    In an agent, the model chooses the next step itself, in a loop, until it answers. In a workflow, your code takes the steps in a fixed order. Anthropic draws the line this way in *Building effective agents* (December 2024): a workflow follows paths that code sets beforehand; an agent directs its own steps.

    The kitchen of this lesson has {dishes} recipes and a pantry. `recipe` gives the ingredients of a dish with their grams, `pantry` how many grams of an ingredient are there:

    ```python
    recipe("pancakes")   # {{"flour": 250, "milk": 450, "eggs": 110, "butter": 25}}
    pantry("milk")       # 200
    pantry("eggs")       # 0
    ```

    Both ways below get the same question and the same two lookups. The agent gets them as tools; the workflow's code calls them itself.
    """.format(dishes=F["dishes"]))
    return


@app.cell(hide_code=True)
def picture_flow():
    mo.Html(C.flow_picture("en"))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### What the course measured

    The course asked Qwen3-0.6B {n} questions, two for each dish, both ways, through the course's server at temperature 0, so that the same request always gets the same answer. The agent got the system prompt of [What an agent is](/course/open/agent-idea?lang=en), which now also asks for the grams of each ingredient, the two tools, and a limit of {steps} requests. The workflow asked the model twice: which dish the question is about, and to write one sentence with the list that the code worked out.
    """.format(n=F["questions"], steps=F["steps"]))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture("en"))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.vstack([
        mo.md(C.measured_text("en")),
        L.watch_out("A workflow only answers the questions it was built for. Ask it *What can I make with what is in my pantry?*, and its first step has nothing to pick: there is no dish in the question. That is why the workflow checks the model's dish before it goes on. An agent could at least try other steps; a workflow should say plainly what it cannot do."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: a workflow or an agent?

    1. **Write the steps down.** If they are the same for every question your app gets, build a workflow.
    2. **Give code what code does exactly:** lookups, arithmetic, checks.
    3. **Give the model what needs language:** reading the question, writing the answer. Check what it fills in before your code goes on.
    4. **Choose an agent only where the next step depends on what the last one found,** in ways you cannot list beforehand, such as finding out why a program fails.
    5. **Measure both on your own questions:** right answers, requests, tokens and seconds.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(C.worked_text("en"))
    return


@app.cell(hide_code=True)
def text_to_buy():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The workflow's own step

        `to_buy(dish)` is the step of the workflow that needs no model. `recipe(dish)` gives the recipe as a dictionary, such as `{"flour": 250, "milk": 450, "eggs": 110, "butter": 25}`, and `pantry(item)` the grams of an ingredient in the pantry, such as `200` for milk. Complete the loop, so that `buy` holds each ingredient the pantry has too little of, with the grams that are missing. The check tries all six dishes; the picture under it shows your list for each.
        """),
        L.task("In the loop, look up the pantry's grams of `item`; when they are fewer than `need[item]`, put `item` in `buy` with the grams that are missing."),
    ])
    return


@app.cell(hide_code=True)
def exercise_to_buy():
    box_to_buy = L.code_box("to-buy", lines=8)
    box_to_buy
    return (box_to_buy,)


@app.cell(hide_code=True)
def check_to_buy(box_to_buy):
    result_to_buy = L.check("to-buy", box_to_buy.value)
    result_to_buy
    return (result_to_buy,)


@app.cell(hide_code=True)
def hints_to_buy():
    L.hints(
        "to-buy",
        [
            "Two amounts side by side: `need[item]` is what the recipe wants, `pantry(item)` what there is. An ingredient is short when there is less than the recipe needs.",
            "Keep what the pantry has in a name, such as `have`, and compare it with `need[item]` in an `if`. The grams to buy are what the recipe needs minus what there is.",
            "`have = pantry(item)`, then `if have < need[item]:`, and under it put the difference in `buy[item]`.",
        ],
        why="""```python
def to_buy(dish):
    need = recipe(dish)
    buy = {}
    for item in need:
        have = pantry(item)
        if have < need[item]:
            buy[item] = need[item] - have
    return buy
```

For pancakes, flour (1000 ≥ 250) and butter (250 ≥ 25) stay off the list; milk gives 450 − 200 = 250, and eggs 110 − 0 = 110. `<` and not `<=`: for tomato soup, the pantry has exactly the 150 grams of onion the recipe needs, so you buy none.""",
    )
    return


@app.cell(hide_code=True)
def text_shopping():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The whole workflow

        `shopping(url, question)` is the workflow of the picture above. Its first line asks the model which dish the question is about, with the prompt `PICK`, and cleans up the answer with `.strip().lower()`. Write the rest: give back `SORRY` when the dish is not one of `DISHES`; otherwise work out what to buy with `to_buy`, ask the model to write the answer with the prompt `WRITE`, and give back what it writes. `WRITE` has two places to fill in: `{dish}`, and `{items}` for the list as text, which `as_text(buy)` gives, such as `"milk 250 g, eggs 110 g"`. The box knows `PICK`, `WRITE`, `DISHES`, `SORRY`, `as_text` and the course's `to_buy`. The check calls your workflow against a stand-in of the server, with a question about pancakes, one about rice pudding, and one about pizza, which the kitchen does not have.
        """),
        L.task("After the first line, give back `SORRY` when the dish is not in `DISHES`; otherwise work out the list with `to_buy`, and give back the model's answer to `WRITE`, filled in with the dish and the list."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shopping():
    box_shopping = L.code_box("shopping", lines=22)
    box_shopping
    return (box_shopping,)


@app.cell(hide_code=True)
def check_shopping(box_shopping):
    result_shopping = L.check("shopping", box_shopping.value)
    result_shopping
    return (result_shopping,)


@app.cell(hide_code=True)
def hints_shopping():
    L.hints(
        "shopping",
        [
            "The model's dish is text from outside your code: check it before you use it. `dish not in DISHES` is `True` when the model picked something the kitchen does not know.",
            "`buy = to_buy(dish)` gives the list. `WRITE.format(dish=dish, items=as_text(buy))` gives the second prompt, and `ask(url, ...)` sends it.",
            "`if dish not in DISHES:` with `return SORRY` under it; then `buy = to_buy(dish)`, and return what `ask` gives for the filled-in `WRITE`.",
        ],
        why="""```python
def shopping(url, question):
    dish = ask(url, PICK.format(question=question)).strip().lower()
    if dish not in DISHES:
        return SORRY
    buy = to_buy(dish)
    return ask(url, WRITE.format(dish=dish, items=as_text(buy)))
```

Two requests, every time, in the same order: that makes a workflow cheap to run and easy to test. The check on `dish` is a gate: the model fills in a step, and your code decides whether the next step may go on.""",
    )
    return


@app.cell(hide_code=True)
def text_agent_task():
    mo.md(r"""
    ### Exercise 3. Where an agent fits
    """)
    return


@app.cell(hide_code=True)
def choice_agent_task_cell():
    choice_agent_task = L.choice(
        "agent-task",
        question="Which of these tasks needs an agent, and not a workflow?",
        options={
            "invoice": "Read each invoice that comes in, and put its date, amount and supplier in a table",
            "translate": "Translate the description of each new product into Dutch and into French",
            "tests": "Find out why the tests of a program fail, and change the code until they pass",
        },
        correct="tests",
        why={
            "tests": "What to do next depends on what the last step found: an error message, a file, a test that still fails. Those steps cannot be listed beforehand.",
            "invoice": "Every invoice takes the same steps: read three fields, check them, add a row. Your code can fix that order; the model only reads.",
            "translate": "Every description takes the same step, twice. A workflow does that, and is cheaper and easier to test than an agent.",
        },
    )
    choice_agent_task
    return (choice_agent_task,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it on Qwen3

    Once exercise 2 is right, you can try your workflow on the real model: the course starts its own server on your computer, with Qwen3-0.6B behind it. Ask about one of the six dishes in your own words, or about a dish the kitchen does not have.
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
def question_box_cell():
    question_box = mo.ui.text(value=C.QUESTIONS[3][0], label="Your question", full_width=True)
    send_button = mo.ui.run_button(label="Ask your workflow")
    mo.vstack([question_box, send_button])
    return question_box, send_button


@app.cell(hide_code=True)
def demo_shopping(box_shopping, question_box, send_button):
    _shopping = C.learner_shopping(box_shopping.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _shopping is None:
        _out = mo.md("*Finish exercise 2 first: the demo asks with your own workflow.*")
    elif not send_button.value:
        _out = mo.md("*Press the button to ask.*")
    elif not question_box.value.strip():
        _out = mo.md("*Write a question in the box first.*")
    else:
        mo.output.replace(mo.md("*The question has gone; the server loads Qwen3 and answers …*"))
        _words = {"title": "What your workflow did with Qwen3", "request": "Request {n}: your workflow asked",
                  "answer": "The server answered:", "code": "Then your code: dish {dish!r}, to buy {buy}",
                  "gave": "Your workflow gave back: {v}", "error": "Your workflow stopped with {e}.",
                  "none": "Your workflow sent nothing to the server.", "status": "The server answered with status {s}: {m}",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_shopping, question_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The loop of [What an agent is: a model, tools and a loop](/course/open/agent-idea?lang=en) and the calls of [Tools as JSON: function calling](/course/open/build-tool-calling?lang=en) let a model choose its own steps. Here your code chose them, and the model filled in two. In the next lesson, [Five workflow patterns](/course/open/build-workflow-patterns?lang=en), you meet the shapes most workflows take: steps in a row, a router, parts side by side, an orchestrator with workers, and an evaluator that sends an answer back.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_agent_task, result_shopping, result_to_buy):
    L.wrap_up(result_to_buy, result_shopping, choice_agent_task)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
