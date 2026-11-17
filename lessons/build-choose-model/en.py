# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-choose-model. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-choose-model", "en")
    C = L.checks   # the course's measurement of two ways on eighteen questions, the router, and the demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Calling a model over HTTP](/course/open/build-call-api?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_usage():
    quick_usage = L.choice(
        "quick-usage",
        question="Where in the JSON `answer` of a server is the number of tokens the model wrote?",
        options={
            "content": "In the length of `answer[\"choices\"][0][\"message\"][\"content\"]`",
            "finish": "In `answer[\"choices\"][0][\"finish_reason\"]`",
            "usage": "In `answer[\"usage\"][\"completion_tokens\"]`",
        },
        correct="usage",
        why={
            "usage": "`usage` counts the tokens: `prompt_tokens` went in, `completion_tokens` came out, and those are what a hosted API bills.",
            "content": "The content is text: its length counts characters, not tokens. Where does the answer count tokens? See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
            "finish": "`finish_reason` says why the model stopped, not how much it wrote. See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
        },
    )
    quick_usage
    return (quick_usage,)


@app.cell(hide_code=True)
def quick_check_model():
    quick_model = L.choice(
        "quick-model",
        question="What does `\"model\"` in the body of a request say?",
        options={
            "which": "Which model at the server should answer",
            "long": "How long the answer may be",
            "who": "Who sends the request",
        },
        correct="which",
        why={
            "which": "A server can have more than one model; `\"model\"` picks one, by the name that `/v1/models` lists.",
            "long": "That is `\"max_tokens\"`. What does `\"model\"` pick? See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
            "who": "The server does not learn that from `\"model\"`. What does it pick? See [Calling a model over HTTP](/course/open/build-call-api?lang=en).",
        },
    )
    quick_model
    return (quick_model,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Hundreds of models answer through an API, from small and cheap to large and dear, and many of them can also think first, which takes longer and costs more. The best-known model is not the best choice for every question: an answer that a small model gets right costs many times as much at a large one, and takes longer. So you choose for your own task: is it right often enough, fast enough, and at what price?

    **An everyday example.** You do not take a taxi to the bakery around the corner, and you do not walk to the airport. The best way to travel depends on the trip, not on which car is fastest. That is only a comparison; below you see the real measurements.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **thinking mode** | the model first writes out its reasoning, as in a chain of thought, and only then the answer; the API gives the reasoning apart from the answer, but counts its tokens as output tokens. A model made to answer this way is called a *reasoning model* |
    | **LLM routing** | sending each question to the model, or the way of answering, that fits it: easy ones to a fast, cheap one, hard ones to a slow, strong one |
    | **router** | the part of your program that does the routing: a function that looks at a question and gives back which model or way gets it |
    | **LLM gateway** | one address in front of many models, at different companies or on your own computers: your code always calls the gateway and names the model in `"model"`; the gateway passes the request on, counts the costs, and can switch to another model when one fails |
    """)
    return


@app.cell(hide_code=True)
def text_measure():
    mo.md(C.measure_text(L.lang))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.vstack([
        mo.md(C.measured_after(L.lang)),
        L.watch_out("These eighteen questions are the course's, not yours: how a model does on them says little about your task. The same goes for a public ranking of models, a leaderboard: it measures the tasks of the ranking. Measure on questions like the ones your own app will get."),
    ])
    return


@app.cell(hide_code=True)
def text_routing():
    mo.md(C.routing_text(L.lang))
    return


@app.cell(hide_code=True)
def picture_router():
    mo.Html(C.router_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_gateway():
    mo.md(r"""
    A router can be a rule, as here, or a small model that has learned to tell easy questions from hard ones. Many apps send their requests through a gateway, one address for all their models (examples in November 2026: LiteLLM, which you run yourself, and OpenRouter, a company that passes your requests on to the models of many others, with one key). Then the router only has to pick a name for `"model"`.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: choose a model

    1. **Collect your own examples:** twenty or more questions like the ones your app will get, each with what a right answer has.
    2. **Mark with code:** a function that says right or wrong, the same for every candidate.
    3. **Ask every candidate several times:** a model samples, so one run says little.
    4. **Count per candidate:** how often it was right, its tokens and seconds, and what a right answer costs.
    5. **Choose:** the cheapest candidate that is right often enough and fast enough; when your questions differ a lot, route them.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(C.worked_text(L.lang))
    return


@app.cell(hide_code=True)
def text_cost():
    _e = C.example(L.lang)
    mo.vstack([
        mo.md(rf"""
        ## Your turn

        ### Exercise 1. What a right answer costs

        Write `cost_per_right(runs, price_out)`: `runs` is a list of runs, each a dictionary with `"right"`, `True` or `False`, and `"tokens"`, the output tokens it used; `price_out` is the price in dollars per million output tokens. Give back what the tokens of all runs cost, divided by the number of right runs; when no run is right, give back `None`. For {_e["n"]} runs with {_e["tokens"]} tokens, of which the one with {_e["wrong"]} was wrong, at {_e["price"]} dollars per million, that is `{_e["want"]}`.
        """),
        L.task("Finish `cost_per_right`: the tokens of all runs, the number of right runs, and `None` when none is right."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cost():
    box_cost = L.code_box("cost_per_right", lines=12)
    box_cost
    return (box_cost,)


@app.cell(hide_code=True)
def check_cost(box_cost):
    result_cost = L.check("cost_per_right", box_cost.value)
    result_cost
    return (result_cost,)


@app.cell(hide_code=True)
def hints_cost():
    L.hints(
        "cost_per_right",
        [
            "Count two things in one loop over `runs`: all tokens, and how many runs are right.",
            "Start with `tokens = 0` and `right = 0`. In the loop, add `run[\"tokens\"]` to `tokens`, and 1 to `right` when `run[\"right\"]` is true. After the loop, give back `None` if `right == 0`.",
            "At the end: `return tokens * ___ / right`.",
        ],
        why="""```python
def cost_per_right(runs, price_out):
    tokens = 0
    right = 0
    for run in runs:
        tokens = tokens + run["tokens"]
        if run["right"]:
            right = right + 1
    if right == 0:
        return None
    return tokens * price_out / 1_000_000 / right
```

The wrong answers cost tokens too. That is why a model that is cheap per token, but often wrong, can cost more per right answer.""",
    )
    return


@app.cell(hide_code=True)
def text_route():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A router

        Write `route(question)`: give back `"think"` when the question has a digit in it, and `"fast"` when it has none. Under the check you see what your router would have done with the course's eighteen questions, with the measured runs of each way.
        """),
        L.task("Finish `route`: `\"think\"` for a question with a digit, `\"fast\"` for any other."),
    ])
    return


@app.cell(hide_code=True)
def exercise_route():
    box_route = L.code_box("route", lines=8)
    box_route
    return (box_route,)


@app.cell(hide_code=True)
def check_route(box_route):
    result_route = L.check("route", box_route.value)
    result_route
    return (result_route,)


@app.cell(hide_code=True)
def hints_route():
    L.hints(
        "route",
        [
            "Go through the question character by character, and stop at the first digit.",
            "`for ch in question:` gives the characters one by one; `in` tells whether a character is in a text, so a text of the ten digits tells whether `ch` is a digit.",
            "In the loop: `if ___:`, with `return \"think\"` one step further in; the answer `\"fast\"` comes only after the loop.",
        ],
        why="""```python
def route(question):
    for ch in question:
        if ch in "0123456789":
            return "think"
    return "fast"
```

A rule is the simplest router: quick, free, and now and then wrong, as for a sum written in words. Whether it pays, you measure.""",
    )
    return


@app.cell(hide_code=True)
def text_support():
    mo.md(r"""
    ### Exercise 3. Which model gets which question?
    """)
    return


@app.cell(hide_code=True)
def choice_support_cell():
    choice_support = L.choice(
        "support",
        question="A web shop's chatbot gets 10,000 questions a day. Most ask for the opening hours or where a parcel is; a few are long complaints that need care. What do you do?",
        options={
            "strong": "Send every question to the strongest model you can get, to be safe",
            "route": "Send the short everyday questions to a small, cheap model and the complaints to a strong one, and measure both on real questions",
            "cheap": "Send every question to the cheapest model; a customer with a wrong answer will ask again",
        },
        correct="route",
        why={
            "route": "The everyday questions are most of them, and a small model handles those for a fraction of the price; the few hard ones get the strong model. Real questions tell whether the router sends them the right way.",
            "strong": "Then 10,000 easy questions a day pay the price of the strongest model. Which questions need it?",
            "cheap": "Then the complaints get the weakest answers, where care counts most. Which questions need more?",
        },
    )
    choice_support
    return (choice_support,)


@app.cell(hide_code=True)
def text_try():
    mo.vstack([
        mo.md(r"""
        ### Your router in front of Qwen3

        Once exercise 2 is right, your `route` picks the way for your question, and the course's server on your computer answers that way. Thinking can take minutes on a laptop: the model writes hundreds of tokens before the answer.
        """),
        mo.md(C.default_question(L.lang)),
    ])
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
    question_box = mo.ui.text(value="What is the capital of Japan?", label="Your question", full_width=True)
    send_button = mo.ui.run_button(label="Route it, and ask Qwen3")
    mo.vstack([question_box, send_button])
    return question_box, send_button


@app.cell(hide_code=True)
def demo_answer(box_route, question_box, send_button):
    _route = C.learner_route(box_route.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _route is None:
        _out = mo.md("*Finish exercise 2 first: the demo picks the way with your own `route`.*")
    elif not send_button.value:
        _out = mo.md("*Press the button to send your question.*")
    elif not question_box.value.strip():
        _out = mo.md("*Write a question in the box first.*")
    else:
        mo.output.replace(mo.md("*Your `route` has picked a way; the server loads Qwen3, and the answer follows …*"))
        _words = {"title_fast": "Qwen3, the normal way", "title_think": "Qwen3, in thinking mode",
                  "done_fast": "Your router chose the normal way: {tokens} output tokens, {seconds} seconds.",
                  "done_think": "Your router chose thinking: {tokens} output tokens with the thought, {seconds} seconds.",
                  "no_answer": "Your router chose thinking, and Qwen3 spent all {tokens} output tokens on its thought, in {seconds} seconds: no answer came. Above is the end of the thought.",
                  "other": "Your route gave {way}, and the demo only knows \"fast\" and \"think\".",
                  "failed": "Your route stopped with an error: {why}",
                  "server": "The server did not answer ({why}). Press the button again.",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_route, question_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The calls are the ones of [Calling a model over HTTP](/course/open/build-call-api?lang=en), with one more setting in the body for thinking, and the tokens they count are what you pay for, as in [Tokens and prices: count before you send](/course/open/build-tokens-prices?lang=en). Where the model you choose runs, on your own computer or at a company, is the next lesson: [Where a model runs](/course/open/build-where-models-run?lang=en).
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_support, result_cost, result_route):
    L.wrap_up(result_cost, result_route, choice_support)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
