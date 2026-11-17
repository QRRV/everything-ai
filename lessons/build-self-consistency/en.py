# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-self-consistency. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-self-consistency", "en")
    C = L.checks   # the course's recorded runs, the checks, and the demo with your ask_many


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Sampling settings through the API](/course/open/build-sampling-api?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_seed():
    quick_seed = L.choice(
        "quick-seed",
        question="You send the same request twice, with temperature 0.7 and the same seed. What comes back?",
        options={
            "same": "The same answer twice, because the seed fixes every random draw",
            "differ": "Two different answers, because a temperature above 0 samples each token",
            "error": "An error: a seed only works with temperature 0",
        },
        correct="same",
        why={
            "same": "The seed fixes the random draws, so the same request with the same seed samples the same tokens. For different answers, each request needs its own seed.",
            "differ": "The temperature makes the model sample, but the seed fixes the draws: the same seed gives the same answer. See [Sampling settings through the API](/course/open/build-sampling-api?lang=en).",
            "error": "A seed works with any temperature; at 0 it changes nothing, because greedy decoding draws nothing. See [Sampling settings through the API](/course/open/build-sampling-api?lang=en).",
        },
    )
    quick_seed
    return (quick_seed,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model that works out a sum step by step can slip in one step, and then the answer at the end is wrong, however good the rest was. With sampling, every run takes a slightly different way. The slips land on different wrong numbers, while the runs that reason well all land on the same right one. So asking the same question several times and taking the number that comes up most often is right more often than one answer. That costs a call per answer, so you want to know how much it brings.

    **An everyday example.** When three people count the same pile of coins and two of them arrive at 85 euros, you trust 85 more than the 79 of the third; their mistakes are unlikely to land on the same number. That is only a comparison; below you measure how often the vote of Qwen3's answers is right.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    _f = C.findings()
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **majority voting** | from a few answers, taking the one that comes up most often |
    | **self-consistency** | asking the same question several times with a temperature above 0, each time with its own seed, and taking the final answer that comes up most often: majority voting over chains of thought |

    ### Twelve sums, nine runs each

    The course asked Qwen3 {questions} sums in words, each {runs} times with its own seed, at the server's temperature 0.7, with this prompt around each question:

    ```text
    {steps}
    ```

    From each answer a small function, `answer_of`, takes the number after *Answer:*. Each row is a sum, with the number of each run, the right number, and the number that came most often:
    """.format(questions=_f["questions"], runs=_f["runs"], steps=C.STEPS))
    return


@app.cell(hide_code=True)
def picture_grid():
    mo.Html(C.grid_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_grid():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        Of all {total} answers, {right} were right. One answer per sum got {k1} of the {questions} sums right, on average. The number that came most often in three answers got {k3} right, in five {k5}, and in all nine {k9}. For 1, 3 and 5, that is the average over every way to pick that many of the nine runs, so the numbers do not hang on which runs came first:
        """.format(total=_f["runs"] * _f["questions"], right=_f["all_right"], questions=_f["questions"],
                   k1=C.number(_f["per_k"][1], L.lang, 1), k3=C.number(_f["per_k"][3], L.lang, 1),
                   k5=C.number(_f["per_k"][5], L.lang, 1), k9=C.number(_f["per_k"][9], L.lang, 0))),
        mo.Html(C.vote_picture(L.lang)),
        mo.md(r"""
        A vote cannot fix a slip that most runs make the same way. For sum {s}, about Lisa's bike, {c} of the nine runs said {v}: that is how many weeks 200 euros take at 15 euros a week when you start from nothing, without the 65 euros Lisa already has. The right answer is {r}, and no run gave it. A vote finds the answer most runs agree on, and that is the right one only when the right way is also the most common way.
        """.format(s=_f["stuck"][0] + 1, c=_f["stuck_vote"][0][1], v=_f["stuck_vote"][0][0], r=C.QUESTIONS[_f["stuck"][0]][1])),
        L.watch_out("A vote needs answers that can be compared exactly, such as a number or the letter of a choice. Two stories or two emails are never the same text, so they cannot be counted; and every answer is a call that you wait for and pay for."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: self-consistency

    1. **Ask for reasoning and a final answer that can be compared,** such as a number on the last line.
    2. **Ask the same question `n` times,** with a temperature above 0 and each time another seed, so that the runs differ.
    3. **Take the final answer from each run,** and leave out the runs that have none.
    4. **Count, and take the answer that comes up most often;** with a tie, take the one that came first.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _f = C.findings()
    _i = _f["example"]
    mo.md(r"""
    ### Worked example

    Sum {k} of the twelve: *{question}*

    1. **Reasoning and a final answer:** the prompt asks to think step by step and to end with *Answer:* and a number.
    2. **The same question nine times:** seeds 0 to 8, at temperature 0.7.
    3. **The final answers:** `answer_of` took {numbers} from the nine runs.
    4. **Count:** {counts}. The most common is {vote}, and the right answer is {right}. One answer alone was right {alone} times out of nine.
    """.format(k=_i + 1, question=C.QUESTIONS[_i][0], numbers=", ".join(str(n) for n in C.numbers(_i)),
               counts=", ".join("{n}: {c}×".format(n=n, c=c) for n, c in _f["example_counts"]),
               vote=C.majority(C.numbers(_i)), right=C.QUESTIONS[_i][1], alone=_f["right"][_i]))
    return


@app.cell(hide_code=True)
def text_majority():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The answer that comes up most often

        Write `majority(answers)`, which gives back the answer that comes up most often in the list `answers`. An answer without a number is `None`; that is no vote, so leave it out. `x is not None` is `True` for every value of `x` that is not `None`. With a tie, the answer that came first wins, and with no votes at all, the function gives back `None`. The box already counts in a dictionary, but gives back the first answer.
        """),
        L.task("Leave out `None`, and give back the answer with the most votes."),
    ])
    return


@app.cell(hide_code=True)
def exercise_majority():
    box_majority = L.code_box("majority", lines=14)
    box_majority
    return (box_majority,)


@app.cell(hide_code=True)
def check_majority(box_majority):
    result_majority = L.check("majority", box_majority.value)
    result_majority
    return (result_majority,)


@app.cell(hide_code=True)
def hints_majority():
    L.hints(
        "majority",
        [
            "Two steps: first count, without `None`; then walk through the counts and remember the answer with the most votes so far.",
            "Put the counting under `if answer is not None:`. Then start with `best = None`, and go through `counts`: an answer takes the place of `best` when `best` is still `None`, or when it has more votes than `best`.",
            "After the counting: `best = None`, then `for answer in counts:` with `if best is None or ___:` and `best = answer` under it. At the end, `return best`.",
        ],
        why="""```python
def majority(answers):
    counts = {}
    for answer in answers:
        if answer is not None:
            if answer in counts:
                counts[answer] = counts[answer] + 1
            else:
                counts[answer] = 1
    best = None
    for answer in counts:
        if best is None or counts[answer] > counts[best]:
            best = answer
    return best
```

A dictionary keeps its keys in the order they came in, so going through `counts` meets the answers in the order of the list. Because a later answer only wins with more votes, `>` and not `>=`, a tie goes to the answer that came first.""",
    )
    return


@app.cell(hide_code=True)
def text_ask():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The same question, `n` times

        The box asks once, with seed 0, and gives back `majority` of that one answer. Change `ask_many(url, question, n)` so that it asks `n` times, each time with another seed, and gives back `majority` of all the numbers. `answer_of` and your `majority` are given. The check runs your function against a stand-in that answers each seed in its own way.
        """),
        L.task("Ask `n` times, each time with another seed, and give back `majority` of the numbers."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ask():
    box_ask = L.code_box("ask_many", lines=18)
    box_ask
    return (box_ask,)


@app.cell(hide_code=True)
def check_ask(box_ask):
    result_ask = L.check("ask_many", box_ask.value)
    result_ask
    return (result_ask,)


@app.cell(hide_code=True)
def hints_ask():
    L.hints(
        "ask_many",
        [
            "Everything from `messages = ...` to `answers.append(...)` has to happen `n` times; only the seed changes each time.",
            "Put those lines in a loop that runs `n` times, four spaces further in, and use its counter as the seed.",
            "Under `answers = []`: `for i in ___:`, with the lines up to and including `answers.append(...)` under it, and `\"seed\": i` in the body; `return majority(answers)` stays after the loop.",
        ],
        why="""```python
import json
import urllib.request


def ask_many(url, question, n):
    answers = []
    for i in range(n):
        messages = [{"role": "user", "content": question}]
        body = {"model": "qwen3-0.6b", "messages": messages, "max_tokens": 300,
                "temperature": 0.7, "seed": i}
        data = json.dumps(body).encode()
        request = urllib.request.Request(url + "/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            answer = json.loads(response.read())
        answers.append(answer_of(answer["choices"][0]["message"]["content"]))
    return majority(answers)
```

Each seed gives its own run, and the same seed gives the same run again: so the same question with the same `n` gives the same vote, which makes a result repeatable.""",
    )
    return


@app.cell(hide_code=True)
def text_worth():
    mo.md(r"""
    ### Exercise 3. When it is worth it
    """)
    return


@app.cell(hide_code=True)
def choice_worth_cell():
    choice_worth = L.choice(
        "worth",
        question="Where does asking five times and voting pay for its five calls?",
        options={
            "story": "For a short story: take the text that came up most often",
            "number": "For a number that has to be right, such as an amount on a price quote",
            "chat": "For every answer in a chat app, since five answers always know more than one",
        },
        correct="number",
        why={
            "number": "A number can be compared exactly, and a wrong one costs more than the four extra calls.",
            "story": "Five runs of a story are five different texts, so no text comes up twice: there is nothing to count. A vote needs answers that can be compared exactly.",
            "chat": "Five calls per answer is five times the waiting and the tokens, and a chat answer is free text that cannot be counted. Keep the vote for answers that can be compared and have to be right.",
        },
    )
    choice_worth
    return (choice_worth,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it on Qwen3

    Once exercises 1 and 2 are right, you can vote on a sum of your own with the real model: the course starts its own server on your computer, with Qwen3-0.6B behind it. The demo asks your question with your `ask_many`, in the prompt of this lesson, and shows the number of each answer and the vote. An answer can take a few hundred tokens, so five answers can take a minute or more. Your computer is not the course's, so your answers can differ.
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
def question_cell():
    question_box = mo.ui.text_area(value=C.QUESTIONS[-1][0], label="Your sum", full_width=True, rows=3)
    n_slider = mo.ui.slider(start=3, stop=9, step=2, value=5, label="Answers", show_value=True)
    vote_button = mo.ui.run_button(label="Ask with your ask_many")
    mo.vstack([question_box, n_slider, vote_button])
    return n_slider, question_box, vote_button


@app.cell(hide_code=True)
def demo_answer(box_ask, box_majority, n_slider, question_box, vote_button):
    _fn = C.learner_ask_many(box_majority.value, box_ask.value)
    _question = question_box.value.strip()
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _fn is None:
        _out = mo.md("*Finish exercises 1 and 2 first: the demo asks with your own `ask_many` and `majority`.*")
    elif not vote_button.value:
        _out = mo.md("*Press the button to ask.*")
    elif not _question:
        _out = mo.md("*Write a sum in the box first.*")
    else:
        mo.output.replace(mo.md("*The question goes out {n} times; the server loads Qwen3 and answers …*".format(n=n_slider.value)))
        _words = {"title": "Your sum, {n} times".format(n=n_slider.value),
                  "run": "Answer {k}: {n}",
                  "vote": "Your ask_many gave back: {got}, in {seconds} seconds.",
                  "failed": "It stopped with an error against the real server: {why}",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_fn, _question, n_slider.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The temperature and seed of [Sampling settings through the API](/course/open/build-sampling-api?lang=en) are what make the runs differ, and what make them repeatable. [What goes into the window](/course/open/build-context-engineering?lang=en) looks at what a request carries with it.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_worth, result_ask, result_majority):
    L.wrap_up(result_majority, result_ask, choice_worth)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
