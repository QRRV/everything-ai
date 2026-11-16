# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-rag-idea. Route: course/route/lessen/bouwen.yaml. Checks, the recorded runs and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-rag-idea", "en")
    C = L.checks   # the questions, the course's recorded runs of Qwen3, the checks, and the demo with your ask


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Why language models make things up](/course/open/lm-hallucination?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_cutoff():
    quick_cutoff = L.choice(
        "quick-cutoff",
        question="A model was trained a year ago. You ask it about a rule your company made last month. What is most likely to happen?",
        options={
            "refuse": "It says it cannot know, because the rule is newer than its training",
            "search": "It looks the rule up on the company's website before it answers",
            "made-up": "It can give an answer that sounds sure but is made up",
        },
        correct="made-up",
        why={
            "made-up": "The model writes likely text, not checked facts, and it knows nothing after its knowledge cutoff. It can still sound sure, so an answer about the new rule may simply be made up.",
            "refuse": "A model does not know what it does not know: it writes likely text, and that can sound sure about a rule it never saw. See [Why language models make things up](/course/open/lm-hallucination?lang=en).",
            "search": "A model on its own looks nothing up: it writes from what it learned, up to its knowledge cutoff. Looking things up first is what this lesson adds. See [Why language models make things up](/course/open/lm-hallucination?lang=en).",
        },
    )
    quick_cutoff
    return (quick_cutoff,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Qwen3-0.6B learned from a great many texts, up to the moment it was trained. It does not know what happened after that, nor what was never in those texts: the handbook of a company, the notes of a team, the numbers this course measured. Ask it anyway, and it answers as if it knew. A company that wants a model to answer questions about its own documents cannot train the model again every time a document changes. There is a simpler way: first look up the text that answers the question, and give it to the model together with the question.

    **An everyday example.** In an open-book exam, you do not have to know every number by heart: you look it up and answer from what the book says. But you have to find the right page, and read what is there instead of what you think should be there. That is only a comparison; below you see what Qwen3 does with and without the right page.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **retrieval** | finding, among many texts, the ones that fit a question |
    | **retrieval-augmented generation** (RAG) | first retrieve texts that fit the question, then put them in the prompt with the question, so that the model can write its answer from them |

    The name comes from a paper by Patrick Lewis and colleagues at Facebook AI in 2020, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. In the same year, REALM from Google let a model retrieve passages from Wikipedia while it learned, and in 2021 RETRO from DeepMind retrieved from a collection of some 2 trillion tokens. Most RAG systems today do it the simple way of this lesson: a search finds the texts, and the prompt carries them to a model that was never trained on them.

    ### What the course measured

    The course asked Qwen3 {n} questions, {seeds} times each, in three ways: without a source; with the point of the course's summaries that answers it; and with the point of another question, a source that does not answer it. Two questions ask what was known long before Qwen3 was trained: the top-5 error of AlexNet in 2012, and the number of hex digits of a SHA-256 checksum. The other six ask what only this course measured. Each square is one answer, and the key under the picture says whether it is right; point at a square to read the answer:
    """.format(n=len(C.QUESTIONS), seeds=C.SEEDS))
    return


@app.cell(hide_code=True)
def picture_ways():
    mo.Html(C.ways_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_ways():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        Without a source, {none} of the {runs} answers were right. Qwen3 knew none of the course's numbers: its answers said that it depends, or explained how you could find out. Even on the two questions from outside the course it was right only once: it gave AlexNet a top-5 error of *{alex1}* in one run and an accuracy of *{alex2}* in another, and a SHA-256 checksum *{sha0}*, *{sha1}*, *{sha2}* and once the right *{sha3}*. With the right source, {right} of {runs} were right. With the source of another question, {wrong} was right. For the course's questions, all {unknown} answers said in some way that the sources do not answer the question, though one of them first read the 929 of the wrong source as the perplexity. For SHA-256, Qwen3 answered from memory, and was wrong three times.
        """.format(none=_f["none"]["right"], runs=_f["none"]["runs"], right=_f["right"]["right"], wrong=_f["wrong"]["right"], unknown=_f["wrong"]["unknown"],
                   alex1="1.14%", alex2="99.9%", sha0="32 hex digits", sha1="64-bit", sha2="64 bits", sha3="64 hex digits")),
        L.watch_out("RAG makes an answer as good as its source. With the right text, Qwen3 was right every time. With a text that does not fit, it mostly said so, but it also took a number from it once, and on a question it thought it knew, it ignored the source. So the search that finds the texts matters as much as the model, and the next lessons are about that search."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: RAG step by step

    1. **Retrieve:** find the texts that fit the question.
    2. **Put them in the prompt:** numbered, with an instruction to use them, and the question after them.
    3. **Generate:** send the prompt to the model, which writes the answer.
    4. **Check:** does the answer say what the source says?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    *{question}*

    1. **Retrieve:** here the course hands over the right text, a point of the summary of the lesson on a small CNN: *{passage}*
    2. **Put it in the prompt:** the instruction, the source with its number, and the question:

        ```text
        {prompt}
        ```

    3. **Generate:** in all {seeds} runs, Qwen3 wrote: *{answer}* Without the source, it began: *{alone}*
    4. **Check:** the source says 9,098 parameters, and so does the answer.
    """.format(question=C.question("q01")["text"]["en"], passage=C.passages_for("q01", "right")[0], seeds=C.SEEDS,
               prompt=C.with_sources(C.question("q01")["text"]["en"], C.passages_for("q01", "right")).replace("\n", "\n    "),
               answer=C.answers("q01", "right")[0]["text"].strip(), alone=C.answers("q01", "none")[0]["text"].split(". ")[0] + "."))
    return


@app.cell(hide_code=True)
def text_sources():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Sources in front of the question

        Write `with_sources(question, passages)`, which makes the prompt of the worked example from a question and a list of passages:

        ```text
        Use these sources to answer the question.

        [1] the first passage
        [2] the second passage

        Question: the question
        ```

        Each passage is on a line of its own, after its number in square brackets and a space. Without passages, the function gives back the question alone. The box puts the passages in front of the question without numbers, and also when there are none. When it is right, the check also keeps it as a file, `work/projects/app/sources.py`: [Project: your assistant reads your documents](/course/open/build-project-rag?lang=en) uses it.
        """),
        L.task("Number the passages from 1, and give back the question alone when there are none."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sources():
    box_sources = L.code_box("with_sources", lines=11)
    box_sources
    return (box_sources,)


@app.cell(hide_code=True)
def check_sources(box_sources):
    result_sources = L.check("with_sources", box_sources.value)
    result_sources
    return (result_sources,)


@app.cell(hide_code=True)
def hints_sources():
    L.hints(
        "with_sources",
        [
            "Two changes: a number that goes up by one with every passage, and a test at the start for a list without passages.",
            "Start a counter at 1 before the loop, put it in front of each passage with an f-string, and add 1 after each one. `len(passages) == 0` is true for an empty list.",
            "Before the loop: `n = 1`. In the loop: `text = text + f\"[{n}] {passage}\\n\"` and `n = n + 1`. At the very start: `if len(passages) == 0:` with `return question` under it.",
        ],
        why="""```python
def with_sources(question, passages):
    if len(passages) == 0:
        return question
    text = "Use these sources to answer the question.\\n\\n"
    n = 1
    for passage in passages:
        text = text + f"[{n}] {passage}\\n"
        n = n + 1
    return text + "\\nQuestion: " + question
```

The numbers let the model, and you, say which source an answer comes from: *according to [2]*. Without passages, an instruction to use sources would only confuse the model.""",
    )
    return


@app.cell(hide_code=True)
def text_ask():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Asking with sources

        The box is the `ask` of [Calling a model over HTTP](/course/open/build-call-api?lang=en), with room for 80 tokens and a third argument, `passages`; it still sends the question alone. Change `ask(url, question, passages)` so that the content of the user's message is the prompt `with_sources` makes of the question and the passages. `with_sources` is given. The check runs your function against a stand-in.
        """),
        L.task("Send the prompt with the sources as the content of the user's message."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ask():
    box_ask = L.code_box("ask", lines=14)
    box_ask
    return (box_ask,)


@app.cell(hide_code=True)
def check_ask(box_ask):
    result_ask = L.check("ask", box_ask.value)
    result_ask
    return (result_ask,)


@app.cell(hide_code=True)
def hints_ask():
    L.hints(
        "ask",
        [
            "Only what goes into the message changes; the request and the answer stay as they are.",
            "Make the prompt first with `with_sources(question, passages)`, and put that in the message instead of `question`.",
            "`prompt = with_sources(question, passages)` before `body`, and in `body` the message `{\"role\": \"user\", \"content\": prompt}`.",
        ],
        why="""```python
import json
import urllib.request


def ask(url, question, passages):
    prompt = with_sources(question, passages)
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": prompt}], "max_tokens": 80}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]
```

The server and the model do not change at all: RAG is only a different prompt.""",
    )
    return


@app.cell(hide_code=True)
def text_needs():
    mo.md(r"""
    ### Exercise 3. When RAG helps
    """)
    return


@app.cell(hide_code=True)
def choice_needs_cell():
    choice_needs = L.choice(
        "needs",
        question="You build an assistant on Qwen3-0.6B. For which of these questions does RAG help the most?",
        options={
            "capital": "What is the capital of France, and which river runs through it?",
            "handbook": "What does Spoke & Chain charge for a full service?",
            "poem": "Write a short poem about autumn leaves for a birthday card.",
        },
        correct="handbook",
        why={
            "handbook": "The price is in the shop's own handbook, which Qwen3 never saw, so on its own it can only guess. With the right part of the handbook in the prompt, it can answer from it.",
            "capital": "Paris and the Seine were in countless texts that Qwen3 learned from, so it can answer without a source. RAG helps most with what the model cannot know, such as the price in the shop's own handbook.",
            "poem": "A poem asks for no fact that a source could give. RAG helps most with what the model cannot know, such as the price in the shop's own handbook.",
        },
    )
    choice_needs
    return (choice_needs,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it on Qwen3

    Once exercises 1 and 2 are right, you can ask Qwen3 the course's questions yourself, with your own `ask`: the course starts its own server on your computer, with Qwen3-0.6B behind it. Choose a question and a source, and ask a few times: the answer can change from one time to the next, and your computer is not the course's.
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
def settings_cell():
    question_choice = mo.ui.dropdown(options={C.question(q)["text"][L.lang]: q for q in C.QUESTIONS},
                                     value=C.question("q01")["text"][L.lang], label="Question")
    way_choice = mo.ui.radio(options={"no source": "none", "the right source": "right", "another question's source": "wrong"},
                             value="the right source", label="Source")
    ask_button = mo.ui.run_button(label="Ask with your ask")
    mo.vstack([question_choice, way_choice, ask_button])
    return ask_button, question_choice, way_choice


@app.cell(hide_code=True)
def demo_answer(ask_button, box_ask, box_sources, question_choice, way_choice):
    _fn = C.learner_ask(box_sources.value, box_ask.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif _fn is None:
        _out = mo.md("*Finish exercises 1 and 2 first: the demo runs your own `with_sources` and `ask`.*")
    elif not ask_button.value:
        _out = mo.md("*Press the button to ask.*")
    else:
        mo.output.replace(mo.md("*The server loads Qwen3, and Qwen3 writes …*"))
        _words = {"title": "Your ask: {question}", "sources": "The sources:", "no_sources": "No sources: the question alone.",
                  "answer": "Qwen3's answer:", "right": "The answer is right; it took {seconds} seconds.",
                  "not": "The answer is not right; it took {seconds} seconds.",
                  "failed": "It stopped with an error against the real server: {why}",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.demo(_fn, question_choice.value, way_choice.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Here the course handed Qwen3 the right source. In [Your first RAG](/course/open/build-rag-first?lang=en), a search finds the sources itself, among all {n} points of the course's summaries, and a system prompt says what the assistant is for.
    """.format(n=C.number(C.point_count(), L.lang, 0)))
    return


@app.cell(hide_code=True)
def wrap_up(choice_needs, result_ask, result_sources):
    L.wrap_up(result_sources, result_ask, choice_needs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
