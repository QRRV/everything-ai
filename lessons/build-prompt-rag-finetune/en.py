# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-prompt-rag-finetune. Route: course/route/lessen/bouwen.yaml. Checks, the bank's replies of both versions and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-prompt-rag-finetune", "en")
    C = L.checks              # the bank's replies of both versions, what the eval set and the course's notes say, and the pictures
    SYSTEM = C.SYSTEM         # the system prompt of version 2: the facts of the six topics and the form of the answer
    QUESTIONS = C.QUESTIONS   # the 120 test questions of the bank
    V1 = C.V1                 # per question, version 1: {"reply", "passed" (the eval set), "notes" (the course's, by hand)}
    V2 = C.V2                 # the same for version 2


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Where a model runs](/course/open/build-where-models-run?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_where():
    quick_where = L.choice(
        "quick-where",
        question="An open-weight model you ran on your own computer now runs at a company that hosts it for you. What changes in your code?",
        options={
            "api": "The form of every request and answer",
            "weights": "You first download the model's weights",
            "address": "The base URL and the model's name",
        },
        correct="address",
        why={
            "address": "The request keeps its form: only the base URL and the model's name change, plus the key a provider asks for. A model that a provider fine-tunes for you is called the same way, under a name of its own.",
            "api": "Local runtimes and providers speak the same API, so requests and answers keep their form. Only the base URL and the model's name change. See [Where a model runs](/course/open/build-where-models-run?lang=en).",
            "weights": "The weights stay at the provider; your code only sends requests. Only the base URL and the model's name change. See [Where a model runs](/course/open/build-where-models-run?lang=en).",
        },
    )
    quick_where
    return (quick_where,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    mo.md(r"""
    ## Why this matters

    Version 2 of the bank's assistant passed the eval set of [An eval set and regression tests](/course/open/build-eval-set?lang=en) for {p2} of the {n} test questions, and the course's notes say it made something up in {m2} replies. What do you change next? There are three ways to change what a model does in your app. You change the prompt. You let the app look up facts and put them in the prompt: RAG, as in [Look it up first: what RAG is](/course/open/build-rag-idea?lang=en). Or you train the model further on your own examples: fine-tuning.

    They fix different things and cost different amounts. A prompt changes in a minute, and the eval set tells you the same hour whether it helped. RAG needs a search over your texts. Fine-tuning needs hundreds of good examples, a training run, and a new model that you keep up to date. Choose wrong, and you spend weeks on a model that still does not know about a new fee.

    **An everyday example.** A new employee keeps giving customers wrong answers. If they lack information, you give them a note with the facts, or the handbook to look things up in. If they lack a habit, you first say clearly what you expect. Only when that does not help do you train them, with examples of good work.
    """.format(p2=_f["passed"]["v2"], n=_f["questions"], m2=_f["made_up"]["v2"]))
    return


@app.cell(hide_code=True)
def text_idea():
    _f = C.findings()
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **fine-tuning** | training a model that is already trained a little further on your own examples, so that its weights change |
    | **hosted fine-tuning** | fine-tuning that a provider does for you: you upload your data, the provider trains, and you call your own version of the model through the same API, under a new name |

    Try the three in a fixed order, and measure each step on your eval set:

    1. **A better prompt** first: instructions, the form of the answer, a few examples, and the facts when there are few. It costs nothing to try and is undone in a minute.
    2. **RAG** when the model lacks facts: too many to fit in the prompt, or facts that change. The bank's facts are {sentences} sentences on {topics} topics, so they fit in the prompt; all the bank's terms and conditions would not.
    3. **Fine-tuning** when the model lacks a behavior that a good prompt does not get reliably, and you have good examples of it. Fine-tuning teaches a way of answering, not facts: a fact learned that way is hard to change, and the model may still make it up.

    The three go together: a fine-tuned model still gets a prompt, and can still look things up.

    A provider fine-tunes in one of three ways, each from a file with one JSON object per line:

    - **examples:** the messages of a chat, ending with the reply you want; the model learns to write replies like it.
    - **preferences:** a question with a better and a worse reply; the model learns to prefer the better one.
    - **a grader:** the file holds questions, and code or a model gives each reply the model writes a score, its reward; the model learns to write replies that score high. Checks like those of your eval set can be a grader, run on questions other than the eval set's.
    """.format(topics=_f["topics"], sentences=_f["fact_sentences"]))
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.flow_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: choose the change

    1. **Measure** on your eval set, and read the failures as in [Look at the mistakes first](/course/open/build-error-analysis?lang=en).
    2. **Ask for each kind of failure:** does the model lack facts, or a behavior?
    3. **Prompt first:** write what you want, with the facts if they fit, and measure again.
    4. **Facts that do not fit, or that change:** RAG.
    5. **A behavior the prompt does not get reliably:** collect good examples from questions other than those of the eval set, fine-tune, and measure on the same eval set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _f = C.findings()
    mo.md(r"""
    ### Worked example

    `QUESTIONS` holds the {n} test questions of the bank. `V1` and `V2` hold, for each question, the reply of version 1 and of version 2 of the prompt, each as a dictionary with `"reply"`, `"passed"`, what the eval set said, and `"notes"`, the kinds of mistakes the course wrote down by hand, such as `"made-up"`. `SYSTEM` is the system prompt of version 2: the facts of the {topics} topics and the form of the answer, {chars} characters.

    A training example for a provider is the chat as the app sends it, with the reply you want at the end. For `QUESTIONS[0]`, *{q0}*, it is the example below: the system prompt, the question with the role `user`, and the reply of version 2 with the role `assistant`, which passed and has no note. The reply is JSON because that is the form the app wants: the model learns the form it sees.
    """.format(n=_f["questions"], topics=_f["topics"], chars=C.number(_f["system_chars"], L.lang), q0=QUESTIONS[0]))
    return


@app.cell(hide_code=True)
def picture_worked_example():
    mo.Html(C.example_picture(0, L.lang))
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    _text = mo.md(r"""
    ### What the bank's replies would give

    The eval set passed {p2} of the replies of version 2. As training examples, those {p2} would teach whatever they show, and the course's notes say {pm} of them made something up: the eval set has no check for that. Kept only when they passed and have no note, {ex} replies are left.

    A preference needs a question where exactly one of the two versions passed. That is {pairs} of the {n} questions: in {v2b}, the reply of version 2 is the better one, in {v1b} the reply of version 1. In {both} questions both passed and in {neither} neither did, so there is no pair: the eval set does not say which reply is better. And in {bm} of the {pairs} pairs, the better reply made something up as well. A preference teaches *more like this*, so look at the better reply too.

    The course does not fine-tune here, and these examples must not go to a provider as they are: they answer the questions of the eval set. A model trained on them can learn the eval set's answers by heart without getting better at new questions. Real examples come from other questions, such as new messages from customers with replies that someone checked.
    """.format(p2=_f["passed"]["v2"], pm=_f["passed_made_up"], ex=_f["examples"], pairs=_f["pairs"], n=_f["questions"], v2b=_f["v2_better"],
               v1b=_f["v1_better"], both=_f["both"], neither=_f["neither"], bm=_f["better_made_up"]))
    mo.vstack([
        _text,
        L.watch_out("Keep the questions of your eval set out of everything you train on. Otherwise the eval set measures what the model has seen, not what it can do."),
    ])
    return


@app.cell(hide_code=True)
def text_examples():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Training examples

        `examples(questions, replies)` should give the training examples a provider takes: for each reply that passed the eval set and has no note, a dictionary `{"messages": [...]}` with three messages: `{"role": "system", "content": SYSTEM}`, the question with the role `"user"`, and the reply with the role `"assistant"`. `replies[i]` is the dictionary of question `i`, such as `V2[i]`, and `r["notes"] == []` is `True` when the course wrote no note. The start code gives every question, without a system prompt and without a reply. The picture runs your function on all 120 replies of version 2.
        """),
        L.task("Keep a reply only when it passed and has no note, and put `SYSTEM` before the question and the reply after it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_examples():
    box_examples = L.code_box("examples", lines=12)
    box_examples
    return (box_examples,)


@app.cell(hide_code=True)
def check_examples(box_examples):
    result_examples = L.check("examples", box_examples.value)
    result_examples
    return (result_examples,)


@app.cell(hide_code=True)
def hints_examples():
    _f = C.findings()
    L.hints(
        "examples",
        [
            "Two things to do: choose which replies to keep, and build the three messages of each example.",
            "Put an `if` before `out.append(...)`; in the list of messages, the system message comes first and the assistant message last.",
            "Make it `if r[\"passed\"] and r[\"notes\"] == []:`, and end the list with `{\"role\": \"assistant\", \"content\": ___}`.",
        ],
        why="""```python
def examples(questions, replies):
    out = []
    for i in range(len(questions)):
        r = replies[i]
        if r["passed"] and r["notes"] == []:
            out.append({{"messages": [
                {{"role": "system", "content": SYSTEM}},
                {{"role": "user", "content": questions[i]}},
                {{"role": "assistant", "content": r["reply"]}},
            ]}})
    return out
```

{pm} of the {p2} replies that the eval set passed made something up. Kept as examples, they would teach the model to make things up too.""".format(pm=_f["passed_made_up"], p2=_f["passed"]["v2"]),
    )
    return


@app.cell(hide_code=True)
def text_pair():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A preference

        `pair(question, a, b)` should turn two replies to one question into a preference, as a provider takes it: `(question, better, worse)`, with the texts of the replies. `a` and `b` are dictionaries like `V1[i]` and `V2[i]`. When exactly one of the two passed the eval set, that one is the better reply. When both or neither passed, give `None`. `not x` is `True` when `x` is `False`. The start code always gives `None`. The picture runs your function on all 120 questions, with `a = V1[i]` and `b = V2[i]`.
        """),
        L.task("Give the better reply's text before the worse one's, and `None` when the eval set does not tell them apart."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pair():
    box_pair = L.code_box("pair", lines=7)
    box_pair
    return (box_pair,)


@app.cell(hide_code=True)
def check_pair(box_pair):
    result_pair = L.check("pair", box_pair.value)
    result_pair
    return (result_pair,)


@app.cell(hide_code=True)
def hints_pair():
    L.hints(
        "pair",
        [
            "Four cases: only `a` passed, only `b` passed, both, or neither. Only the first two make a pair.",
            "Ask `if a[\"passed\"] and not b[\"passed\"]:` for the first case and a second `if` the other way round; after both, `return None`.",
            "In the second `if`: `return (question, ___, a[\"reply\"])`.",
        ],
        why="""```python
def pair(question, a, b):
    if a["passed"] and not b["passed"]:
        return (question, a["reply"], b["reply"])
    if b["passed"] and not a["passed"]:
        return (question, b["reply"], a["reply"])
    return None
```

When both passed, the eval set does not say which reply is better, so a pair there would teach a preference that no one measured.""",
    )
    return


@app.cell(hide_code=True)
def text_first():
    mo.md(r"""
    ### Exercise 3. What first?
    """)
    return


@app.cell(hide_code=True)
def choice_first_cell():
    choice_first = L.choice(
        "first-change",
        question="The bank starts charging 2.50 euros for a card payment abroad, and its assistant must say so. What do you change first?",
        options={
            "tune": "Fine-tune on replies that name the fee",
            "facts": "Add it to the facts in the prompt",
            "rag": "Build RAG over all the bank's pages",
        },
        correct="facts",
        why={
            "facts": "The facts are in the prompt already, a few sentences per topic: one more sentence, and the next reply can name the fee. Measure on the eval set the same day. RAG comes when the facts no longer fit in the prompt; fine-tuning is for a behavior, not for a fact that can change.",
            "tune": "A fine-tuned model learns a fact only from many examples, may still make it up, and keeps the old fee when the fee changes again. A fact belongs in the prompt or in a search: here, one sentence in the facts of the prompt.",
            "rag": "The facts of {topics} topics, a few sentences each, fit easily in the prompt. RAG pays off when the facts no longer fit, such as all the terms and conditions; here, add one sentence to the facts.".format(topics=C.findings()["topics"]),
        },
    )
    choice_first
    return (choice_first,)


@app.cell(hide_code=True)
def text_leak():
    mo.md(r"""
    ### Exercise 4. Which examples?
    """)
    return


@app.cell(hide_code=True)
def choice_leak_cell():
    choice_leak = L.choice(
        "eval-leak",
        question="Why should the bank not fine-tune on the replies to the 120 questions of its eval set?",
        options={
            "leak": "The eval set would test what it taught",
            "few": "120 replies are too few",
            "json": "Training cannot use replies written as JSON",
        },
        correct="leak",
        why={
            "leak": "A model trained on the eval set's questions and replies can learn them by heart, without getting better at new questions. Keep the eval set apart, and take examples from other questions.",
            "few": "The number is not the problem here: these are the eval set's own questions. A model trained on them can learn the eval set's answers by heart, without getting better at new questions.",
            "json": "The app wants JSON, so the examples should be JSON too: the model learns the form it sees. The problem is that these are the eval set's own questions: trained on them, the model learns the eval set's answers by heart.",
        },
    )
    choice_leak
    return (choice_leak,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Every step is measured with the eval set of [An eval set and regression tests](/course/open/build-eval-set?lang=en), on the failures that [Look at the mistakes first](/course/open/build-error-analysis?lang=en) found. RAG starts with [Look it up first: what RAG is](/course/open/build-rag-idea?lang=en), and [Is the RAG right?](/course/open/build-rag-eval?lang=en) measures it. A fine-tuned model is called like any other model, as in [Where a model runs](/course/open/build-where-models-run?lang=en). What fine-tuning does inside the model, you see in [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en) and [DPO: preferences without a reward model](/course/open/llm-dpo-loss?lang=en).
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_first, choice_leak, result_examples, result_pair):
    L.wrap_up(result_examples, result_pair, choice_first, choice_leak)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
