# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-workflow-patterns. Route: course/route/lessen/bouwen.yaml. Checks, the course's runs, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-workflow-patterns", "en")
    C = L.checks   # the five patterns, the course's runs of Qwen3 (tips under a word limit, menus split into dishes), the checks and the demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [A workflow or an agent?](/course/open/build-workflow-or-agent?lang=en), and one about [Choosing a model, and routing between models](/course/open/build-choose-model?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_gate():
    quick_gate = L.choice(
        "quick-gate",
        question="In the workflow `shopping`, the model names the dish. What does the code do with that name before it calls `to_buy`?",
        options={
            "check": "It checks that the dish is one of `DISHES`, and gives back `SORRY` if it is not",
            "trust": "Nothing: the model read the question, so the dish it names is the one that was asked about",
            "again": "It asks the model a second time, and goes on only when both answers name the same dish",
        },
        correct="check",
        why={
            "check": "The model fills in a step; the code decides whether the next step may go on. That check is a gate.",
            "trust": "A small model can name a dish the kitchen does not have, or a word that is no dish at all. See [A workflow or an agent?](/course/open/build-workflow-or-agent?lang=en).",
            "again": "Asking again costs a request and can give the same wrong name twice. A check in code is exact and free. See [A workflow or an agent?](/course/open/build-workflow-or-agent?lang=en).",
        },
    )
    quick_gate
    return (quick_gate,)


@app.cell(hide_code=True)
def quick_check_router():
    quick_router = L.choice(
        "quick-router",
        question="What does a router between a small and a large model do?",
        options={
            "both": "It sends every question to both models, and keeps the longer answer",
            "train": "It trains the small model on the large model's answers, so that in the end one model is enough",
            "send": "It sends each question to the model that suits it, such as easy ones to the small one",
        },
        correct="send",
        why={
            "send": "The router decides per question which path it takes. In this lesson, the paths can also be prompts or pieces of code.",
            "both": "That costs both models every time. A router picks one. See [Choosing a model](/course/open/build-choose-model?lang=en).",
            "train": "That is a different idea. A router leaves both models as they are, and only chooses. See [Choosing a model](/course/open/build-choose-model?lang=en).",
        },
    )
    quick_router
    return (quick_router,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The workflow of the last lesson had three steps in a row: the model names the dish, your code works out the list, the model writes the answer. Not every task fits in a row. A question can be one of several kinds; a menu can hold two dishes or five; a tip can come out too long. Most workflows that people build are made of a few shapes that answer exactly these cases. Once you know them, you see which one a task needs, and you build it in a few lines.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **orchestrator-workers** | a pattern in which the model first decides which parts a task has (the orchestrator), and each part goes to its own call, a worker; then your code puts the results together |
    | **evaluator-optimizer** | a pattern in which one call writes an answer, a check judges it, and when the answer is not good, it goes back to the writer with what is wrong, in a loop, until it passes or the rounds run out |
    | **worker** | the call, or the function, that does one part of the task |

    Anthropic describes five patterns in *Building effective agents* (December 2024). Here they are on the kitchen of the last lesson:

    1. **Steps in a row** ([prompt chaining](/course/open/start-prompt-chains?lang=en)): each step reads what the last one gave. Between two steps, your code can check the result before it goes on, as `shopping` did with the dish.
    2. **Router**: a first call decides what kind of question it is, and each kind gets its own path: a question about a recipe, one about shopping, one the kitchen cannot answer. It is the router of [Choosing a model](/course/open/build-choose-model?lang=en), but the paths can be prompts or code too.
    3. **Parts side by side**: parts that do not need each other run at the same time, such as a tip for each of the six dishes; or the same question goes out three times, and your code keeps the answer that comes back most often.
    4. **Orchestrator and workers**: like parts side by side, but the parts are not known beforehand. For *What do I buy for pancakes and tomato soup?* the model first names the dishes; your code gives each dish to a worker, and puts the lists together.
    5. **Evaluator and optimizer**: an answer, a check, and back with what is wrong. The check can be another call of a model, or code when there is a clear rule, such as a limit on words.
    """)
    return


@app.cell(hide_code=True)
def picture_patterns():
    mo.Html(C.patterns_picture("en"))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### What the course measured

    The course tried two of the patterns on Qwen3-0.6B, at temperature 0, so that the same request always gets the same answer. For the evaluator, it asked for a tip for each of the {dishes} dishes, in at most {most} words. A check in code counted the words, and a tip that was too long went back to the model, up to {tries} answers per dish. The course tried two feedback texts. The first: *{again}* The second: *{new}*
    """.format(dishes=F["dishes"], most=F["limit"], tries=F["tries"],
               again=C.AGAIN.format(n="…", limit=F["limit"]), new=C.FEEDBACK.format(n="…", limit=F["limit"])))
    return


@app.cell(hide_code=True)
def picture_tips():
    mo.Html(C.tips_picture("en"))
    return


@app.cell(hide_code=True)
def text_after_tips():
    mo.md(C.tips_text("en"))
    return


@app.cell(hide_code=True)
def picture_plans():
    mo.Html(C.plans_picture("en"))
    return


@app.cell(hide_code=True)
def text_after_plans():
    mo.vstack([
        mo.md(C.plans_text("en")),
        L.watch_out("An evaluator is only as good as its check. A check in code, such as counting words or reading JSON, says exactly what is wrong. When a model is the judge, it can be wrong too, and a loop that waits for its *yes* may never end. Give every loop a limit on its rounds, and decide what your app does when no answer passes."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: which pattern?

    1. **Are the steps the same for every input, one after the other?** Steps in a row, with a check in code between two steps where a mistake would carry on.
    2. **Do different kinds of input need different handling?** A router, with a path for input it does not know.
    3. **Are there parts that do not need each other, and do you know them beforehand?** Parts side by side.
    4. **Do you only know the parts once you have read the input?** An orchestrator that names them, and a worker per part; check every part it names.
    5. **Is there a clear test for a good answer?** An evaluator, with a limit on the rounds.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(C.worked_text("en"))
    return


@app.cell(hide_code=True)
def text_improve():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The evaluator's loop

        `improve(ask, task, check, rounds)` is the evaluator and optimizer. `ask(messages)` sends the messages to the model and gives back its answer as text; `check(answer)` gives `None` when the answer is good, or a text that says what is wrong, such as `"Your tip has 15 words. Write it again in at most 8 words."`. Complete the loop. The model only writes a better answer when it reads its own answer and what was wrong with it. The check tries five stand-ins of the model: one that is right at once, ones that need two or three rounds, and ones that never get it right in time. The picture under it follows your loop round by round.
        """),
        L.task("In the loop, give back the answer when the check finds no problem; otherwise add the answer and the problem to `messages`, each with the role that the model needs to read it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_improve():
    box_improve = L.code_box("improve", lines=10)
    box_improve
    return (box_improve,)


@app.cell(hide_code=True)
def check_improve(box_improve):
    result_improve = L.check("improve", box_improve.value)
    result_improve
    return (result_improve,)


@app.cell(hide_code=True)
def hints_improve():
    L.hints(
        "improve",
        [
            "Two cases after `problem = check(answer)`: no problem, and a problem. In the first, the loop is done. In the second, the next round must know what went wrong.",
            "Both go into `messages` as dictionaries like the first one: the model's own answer, then what is wrong with it, each with the role of whoever said it. Then the loop asks again.",
            "After the `if`: `messages = messages + [{\"role\": ___, \"content\": answer}, {\"role\": ___, \"content\": problem}]`.",
        ],
        why="""```python
def improve(ask, task, check, rounds):
    messages = [{"role": "user", "content": task}]
    for n in range(rounds):
        answer = ask(messages)
        problem = check(answer)
        if problem is None:
            return answer
        messages = messages + [{"role": "assistant", "content": answer},
                               {"role": "user", "content": problem}]
    return None
```

The `None` after the loop matters: when no answer passed in time, your code knows it, and can say so instead of using a tip that is too long.""",
    )
    return


@app.cell(hide_code=True)
def text_orchestrate():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The orchestrator

        `orchestrate(ask, question, worker)` is an orchestrator with workers. Its first line asks the model which dishes the question is about, with the prompt `PLAN`, and reads the JSON it gives back, such as `{"dishes": ["pancakes", "tomato soup"]}`. `worker(dish)` gives the shopping list of one dish as text, such as `"pancakes: milk 250 g, eggs 110 g"`. Complete the function, so that `results` holds the worker's list for each dish of the plan, in the plan's order. The plan comes from a model: a dish the kitchen does not know has no list. The box knows `PLAN` and `DISHES`. The check tries plans of two and three dishes, one with a pizza in it, and an empty plan.
        """),
        L.task("Give each dish of the plan that is in `DISHES` to `worker`, and add what it gives back to `results`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_orchestrate():
    box_orchestrate = L.code_box("orchestrate", lines=10)
    box_orchestrate
    return (box_orchestrate,)


@app.cell(hide_code=True)
def check_orchestrate(box_orchestrate):
    result_orchestrate = L.check("orchestrate", box_orchestrate.value)
    result_orchestrate
    return (result_orchestrate,)


@app.cell(hide_code=True)
def hints_orchestrate():
    L.hints(
        "orchestrate",
        [
            "The plan decides how many workers there are: one per dish in `plan[\"dishes\"]`. A loop over that list does it, however long it is.",
            "In the loop, skip a dish the kitchen does not know; give every other dish to `worker`, and append its result to `results`.",
            "`for dish in plan[\"dishes\"]:`, then `if ___:`, then `results.append(worker(dish))`.",
        ],
        why="""```python
def orchestrate(ask, question, worker):
    plan = json.loads(ask(PLAN.format(question=question)))
    results = []
    for dish in plan["dishes"]:
        if dish in DISHES:
            results.append(worker(dish))
    return results
```

The model decides the parts; your code hands them out and checks each one. The workers do not need each other, so they could also run at the same time.""",
    )
    return


@app.cell(hide_code=True)
def text_pattern():
    mo.md(r"""
    ### Exercise 3. The pattern for a task
    """)
    return


@app.cell(hide_code=True)
def choice_pattern_cell():
    choice_pattern = L.choice(
        "pattern",
        question="A shop has a model translate each product description into Dutch. A script can check whether every number and product name of the original is still in the translation. Which pattern fits?",
        options={
            "router": "A router that sends long descriptions to a large model, and short ones to a small model",
            "evaluator": "An evaluator: a translation that loses a number or a name goes back, with what is missing",
            "orchestrator": "An orchestrator that first splits each description into sentences, with a worker per sentence",
        },
        correct="evaluator",
        why={
            "evaluator": "There is a clear test for a good translation, and the script can say exactly what is missing. That is what an evaluator needs, with a limit on the rounds.",
            "router": "A router can save money, but nothing in it notices a translation that lost a number.",
            "orchestrator": "Splitting into sentences does not catch a lost number either, and the parts of a description are known beforehand.",
        },
    )
    choice_pattern
    return (choice_pattern,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it on Qwen3

    Once exercise 1 is right, you can try your loop on the real model: the course starts its own server on your computer, with Qwen3-0.6B behind it, and the course's check counts the words. Choose a dish, and see how many rounds your `improve` needs, up to {tries}.
    """.format(tries=F["tries"]))
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
def dish_box_cell():
    dish_box = mo.ui.dropdown(options=list(C.DISHES), value=C.DISHES[1], label="Dish")
    send_button = mo.ui.run_button(label="Ask with your improve")
    mo.vstack([dish_box, send_button])
    return dish_box, send_button


@app.cell(hide_code=True)
def demo_improve(box_improve, dish_box, send_button):
    _improve = C.learner_improve(box_improve.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _improve is None:
        _out = mo.md("*Finish exercise 1 first: the demo asks with your own loop.*")
    elif not send_button.value:
        _out = mo.md("*Press the button to ask.*")
    else:
        mo.output.replace(mo.md("*The question has gone; the server loads Qwen3 and answers …*"))
        _words = {"title": "Your improve with Qwen3, round by round", "round": "Round {n}: the model wrote",
                  "ok": "good", "gave": "Your improve gave back: {v}", "error": "Your improve stopped with {e}.",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_improve, dish_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The workflow of [A workflow or an agent?](/course/open/build-workflow-or-agent?lang=en) was steps in a row with a check between them; the router of [Choosing a model](/course/open/build-choose-model?lang=en) became one of the five patterns. In all five, your code decides the paths. In the next lesson, [The agent loop](/course/open/build-agent-loop?lang=en), the model decides them again: you build the whole loop of an agent through the API, with a thought before each tool call.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_pattern, result_improve, result_orchestrate):
    L.wrap_up(result_improve, result_orchestrate, choice_pattern)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
