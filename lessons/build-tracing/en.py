# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-tracing. Route: course/route/lessen/bouwen.yaml. Checks, the traces of the kitchen agent and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-tracing", "en")
    C = L.checks   # the twelve runs of the kitchen agent as traces, and the pictures
    TRACE = C.TRACE
    TRACES = C.TRACES
    NEEDED = C.Named(C.needed(C.EXAMPLE), "NEEDED")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [The agent loop](/course/open/build-agent-loop?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_loop():
    quick_loop = L.choice(
        "quick-loop",
        question="When does the loop of an agent stop?",
        options={
            "thought": "When the model writes a line in its answer that starts with Thought:",
            "calls": "When an answer has no tool calls, or at the step limit",
            "tool": "When a tool sends back its result",
        },
        correct="calls",
        why={
            "calls": "An answer without tool calls is the final answer, and the step limit stops an agent that keeps calling tools. Every turn of that loop is a call to the model: in this lesson, each one is a span.",
            "thought": "A thought is text the model writes before a tool call; the loop goes on and runs the call. See [The agent loop](/course/open/build-agent-loop?lang=en).",
            "tool": "After a result the loop asks the model again, and the model decides whether it needs another tool. See [The agent loop](/course/open/build-agent-loop?lang=en).",
        },
    )
    quick_loop
    return (quick_loop,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    mo.md(r"""
    ## Why this matters

    A customer of the kitchen app asks what to buy for pasta with pesto, waits {s} seconds, and gets a sentence about what the agent still has to check. The answer alone tells you neither why it took so long nor where it went wrong. The agent asked the model {c} times and made {tn} tool calls in between; what each step got and gave is gone once the request is over, unless your code wrote it down.

    **An everyday example.** A parcel service writes down every stop of a parcel: picked up, sorted, on the van, delivered, each with its time. When a parcel is late, the list shows where it waited; when it went to the wrong address, the list shows at which stop.
    """.format(s=C.sec(_f["ex_seconds"], L.lang), c=_f["ex_calls"], tn=len(_f["ex_tools"])))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **span** | one step of a request, written down: its name, when it started and ended, what went in and what came out, and the span it ran in |
    | **trace** | all spans of one request, nested: the request is the outer span, and each call to the model and each tool call is a span in it; also called a request trace |
    | **observability** | being able to see afterwards what your app did and why, from what it wrote down while it ran, such as traces |

    Tools for traces often write the spans in the form of OpenTelemetry, a shared standard, so that any tool can read the traces of any app; it has agreed names for the parts of a call to a model, such as its tokens. In deep learning frameworks, tracing means something else: recording the operations of a model to make it run faster.

    A trace answers three questions that the answer alone cannot: where did the time go, what did the request cost, and at which step did it go wrong. You only have it when your code writes the spans down while it runs, for every request: you do not know in advance which one a customer will ask about.
    """)
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: a trace of a request

    1. **Span:** around each step, a call to the model or a tool, write down its name, its start and end, and what went in and came out; for a call to the model, its tokens too.
    2. **Nest:** give each span the id of the span it runs in. The request is the outer span.
    3. **Keep:** store the spans of every request, also the ones that went well.
    4. **Read:** the slowest spans for the time, the tokens of the calls for the cost, and what went in and came out, span by span, for where it went wrong.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The course wrote the twelve runs of the kitchen agent with ReAct from [The agent loop](/course/open/build-agent-loop?lang=en) as traces. A span is a dictionary: `"id"`, `"parent"` (the id of the span it runs in, `None` for the request), `"name"`, `"start"` and `"end"` in seconds since the request came in, for a call to the model `"prompt_tokens"` and `"output_tokens"`, for a tool `"input"`, its arguments, and `"output"`, what came out. `TRACE` is run {i}, pasta with pesto, a list of {k} spans:
    """.format(i=C.EXAMPLE, k=len(C.TRACE)))
    return


@app.cell(hide_code=True)
def picture_spans():
    mo.Html(C.spans_table(C.EXAMPLE, L.lang))
    return


@app.cell(hide_code=True)
def text_read():
    _f = C.findings()
    _t = _f["ex_call_seconds"]
    mo.md(r"""
    1. **Time:** the request took {s} seconds. Its {c} calls to the model took {t0}, {t1}, {t2} and {t3} seconds. Its {tn} tool calls last {z} seconds in the trace: the recording timed only the calls to the model, and the kitchen's tools are Python functions that answer in well under a millisecond.
    2. **Cost:** each call to the model got the whole conversation again, so its prompt grew from {p0} to {p3} tokens. Together the calls got {ps} prompt tokens and wrote {os}: {t} tokens for a question of {w} words.
    3. **Where it went wrong:** the recipe has four ingredients. The agent asked the pantry about {a0} and {a1}, never about {m0} and {m1}. Its last call wrote a thought and called no tool, so the loop stopped, and the customer got that thought as the answer.
    """.format(s=C.sec(_f["ex_seconds"], L.lang), c=_f["ex_calls"], t0=C.sec(_t[0], L.lang), t1=C.sec(_t[1], L.lang), t2=C.sec(_t[2], L.lang), t3=C.sec(_t[3], L.lang),
               p0=_f["ex_prompts"][0], p3=_f["ex_prompts"][-1], ps=C.thousands(_f["ex_prompt_sum"], L.lang), os=sum(_f["ex_outputs"]),
               t=C.thousands(_f["ex_tokens"], L.lang), w=_f["ex_question_words"], a0=_f["ex_tools"][1][1]["item"], a1=_f["ex_tools"][2][1]["item"],
               m0=_f["ex_missed"][0], m1=_f["ex_missed"][1], tn=len(_f["ex_tools"]), z=C.sec(0.0, L.lang)))
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    mo.md(r"""
    ### What the course measured

    In the twelve traces together, {mc} calls to the model took {ms} seconds; the {tc} tool calls, well under a millisecond each, hardly count. The slowest span was the first call of run {sr}: {ss} seconds, in which the model wrote {so} tokens. Every first call got {fp0} to {fp1} prompt tokens, the system prompt, the tools and the question, and each later call more. In {np} of the {n} runs the agent never asked the pantry anything, and in every run at least one ingredient was never looked up. Run 5 gave the only right answer, and it never looked up milk, cheese or butter: the pantry happened to have enough. A right answer can hide a wrong way to it; the trace shows the way.
    """.format(mc=_f["model_calls"], ms=C.sec(_f["model_seconds"], L.lang), tc=_f["tool_calls"],
               sr=_f["slowest_run"], ss=C.sec(_f["slowest_seconds"], L.lang), so=_f["slowest_output"], fp0=_f["first_prompt"][0], fp1=_f["first_prompt"][1],
               np=len(_f["never_pantry"]), n=_f["n"]))
    return


@app.cell(hide_code=True)
def picture_measured():
    L.watch_out("A trace holds what customers wrote and what the model answered. Keep it as safely as the chats themselves, and only as long as you need it.")
    return


@app.cell(hide_code=True)
def text_slowest():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The slowest span

        `slowest(spans)` should give the span in the request that took longest: of the spans with `span["parent"] == 0`, the one whose `span["end"] - span["start"]` is the largest. The start code keeps the last span in the request, whatever it took. The picture under the check shows your span in each of the twelve traces, `TRACES`.
        """),
        L.task("Keep a span only when it took longer than `best`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_slowest():
    box_slowest = L.code_box("slowest", lines=8)
    box_slowest
    return (box_slowest,)


@app.cell(hide_code=True)
def check_slowest(box_slowest):
    result_slowest = L.check("slowest", box_slowest.value)
    result_slowest
    return (result_slowest,)


@app.cell(hide_code=True)
def hints_slowest():
    L.hints(
        "slowest",
        [
            "How long a span took is its end minus its start. Compare that of each span with that of `best`, the slowest so far.",
            "A second `if` inside the first: when `span[\"end\"] - span[\"start\"]` is larger than the same for `best`, the span becomes `best`.",
            "Under `if span[\"parent\"] == 0:`, write `if span[\"end\"] - span[\"start\"] > ___:`, with `best = span` one step further in.",
        ],
        why="""```python
def slowest(spans):
    best = spans[1]
    for span in spans:
        if span["parent"] == 0:
            if span["end"] - span["start"] > best["end"] - best["start"]:
                best = span
    return best
```

The request span is left out: it lasts as long as everything in it, so it would always be the slowest.""",
    )
    return


@app.cell(hide_code=True)
def text_tokens():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The tokens of a request

        `tokens_of(spans)` should give the tokens of all calls to the model in a request: for each span with the name `"model"`, its `"prompt_tokens"` plus its `"output_tokens"`, added up. The start code counts only the output tokens.
        """),
        L.task("Add the prompt tokens of each call as well."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tokens():
    box_tokens = L.code_box("tokens_of", lines=7)
    box_tokens
    return (box_tokens,)


@app.cell(hide_code=True)
def check_tokens(box_tokens):
    result_tokens = L.check("tokens_of", box_tokens.value)
    result_tokens
    return (result_tokens,)


@app.cell(hide_code=True)
def hints_tokens():
    L.hints(
        "tokens_of",
        [
            "A call to the model costs the tokens it got, the prompt, and the tokens it wrote.",
            "In the line inside the `if`, add the prompt tokens of the span to what the line already adds.",
            "Make that line `total = total + ___ + span[\"output_tokens\"]`.",
        ],
        why="""```python
def tokens_of(spans):
    total = 0
    for span in spans:
        if span["name"] == "model":
            total = total + span["prompt_tokens"] + span["output_tokens"]
    return total
```

With a price per token, this number is what a request costs. The prompt tokens are most of it: every call sends the whole conversation again.""",
    )
    return


@app.cell(hide_code=True)
def text_looked():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. What the agent never looked up

        `not_looked_up(spans, needed)` should give the items of `needed`, in their order, that no span of the pantry asked about. The start code already collects in `asked` the item of each pantry span, from its input. `x not in a_list` is `True` when `x` does not occur in the list. `NEEDED` holds the ingredients of pasta with pesto.
        """),
        L.task("Go through `needed`, and keep each item that is not in `asked`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_looked():
    box_looked = L.code_box("not_looked_up", lines=10)
    box_looked
    return (box_looked,)


@app.cell(hide_code=True)
def check_looked(box_looked):
    result_looked = L.check("not_looked_up", box_looked.value)
    result_looked
    return (result_looked,)


@app.cell(hide_code=True)
def hints_looked():
    L.hints(
        "not_looked_up",
        [
            "The answer is a list of items from `needed`: the ones the agent never asked about.",
            "A loop `for item in needed:` with an `if` inside it that asks whether `item` is missing from `asked`.",
            "Under `for item in needed:`, write `if ___:`, with `missing.append(item)` one step further in.",
        ],
        why="""```python
def not_looked_up(spans, needed):
    asked = []
    for span in spans:
        if span["name"] == "pantry":
            asked.append(span["input"]["item"])
    missing = []
    for item in needed:
        if item not in asked:
            missing.append(item)
    return missing
```

The step where a run went wrong tells you what to check next: an input like this one belongs in your eval set.""",
    )
    return


@app.cell(hide_code=True)
def text_faster():
    _f = C.findings()
    mo.md(r"""
    ### Exercise 4. Make it faster

    Run {i} took {s} seconds.
    """.format(i=C.EXAMPLE, s=C.sec(_f["ex_seconds"], L.lang)))
    return


@app.cell(hide_code=True)
def choice_faster_cell():
    _f = C.findings()
    choice_faster = L.choice(
        "faster",
        question="What would make run {i} faster?".format(i=C.EXAMPLE),
        options={
            "tool": "A faster pantry tool",
            "question": "A shorter question from the customer",
            "calls": "Fewer or shorter calls to the model",
        },
        correct="calls",
        why={
            "calls": "Its {c} calls to the model took all {s} seconds; its tools, well under a millisecond each, hardly count. A call takes longer the more it writes: about {pt} seconds for each token it wrote, on the course's machine. Fewer steps, or shorter thoughts, make it faster.".format(
                c=_f["ex_calls"], s=C.sec(_f["ex_seconds"], L.lang), pt=C.sec(round(_f["per_output_token"], 2), L.lang)),
            "tool": "The pantry tool answers in well under a millisecond: the kitchen's tools are Python functions. All {s} seconds went to the {c} calls to the model.".format(
                s=C.sec(_f["ex_seconds"], L.lang), c=_f["ex_calls"]),
            "question": "The question has {w} words. The first call got {p} prompt tokens, most of them the system prompt and the tools, and each later call got all of it again. The time went to the calls to the model, not to the question.".format(
                w=_f["ex_question_words"], p=_f["ex_prompts"][0]),
        },
    )
    choice_faster
    return (choice_faster,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Read the traces yourself

    Choose a run, and read its trace: the waterfall for the time, and the spans for what went in and came out.
    """)
    return


@app.cell(hide_code=True)
def settings_run():
    run_pick = mo.ui.dropdown(options=C.run_options(L.lang), value=list(C.run_options(L.lang))[C.EXAMPLE], label="Run", full_width=True)
    run_pick
    return (run_pick,)


@app.cell(hide_code=True)
def demo_trace(run_pick):
    mo.vstack([mo.Html(C.trace_picture(run_pick.value, L.lang)), mo.Html(C.spans_table(run_pick.value, L.lang))])
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** These are the runs you read as thoughts, actions and observations in [The agent loop](/course/open/build-agent-loop?lang=en), now with their time and tokens. An eval set tells you that an answer is wrong; its trace tells you where. In [Guardrails](/course/open/build-guardrails?lang=en), each check before and after the model gets a span of its own, and [How agents go wrong](/course/open/build-agent-failures?lang=en) counts the kinds of mistake in traces like these.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_faster, result_looked, result_slowest, result_tokens):
    L.wrap_up(result_slowest, result_tokens, result_looked, choice_faster)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
