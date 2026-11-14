# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-research-and-learn. Route: course/route/lessen/bouwen.yaml. Checks, the run of Qwen3 as a tutor and its picture: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import chat
    from everything_ai.lesson import lesson

    L = lesson("build-research-and-learn", "en")
    C = L.checks   # the course's run of Qwen3 as a tutor, what each answer did, and the picture


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [AI at work: centaur or cyborg](/course/open/build-ai-at-work?lang=en), and one about [Learning with your own AI assistant](/course/open/start-your-own-assistant?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_frontier():
    quick_frontier = L.choice(
        "quick-frontier",
        question="You want to use an assistant for a new kind of task. How do you find out whether it does that task well?",
        options={
            "ask": "Ask the assistant whether it is good at it",
            "try": "Try it on a few cases where you know the right answer",
            "size": "A large model does every task well, so there is no need",
        },
        correct="try",
        why={
            "try": "The frontier is jagged: you find out per task, by trying it on cases you can check.",
            "ask": "What a model says about itself is text like any other answer. How did you find the frontier of Qwen3? See [AI at work: centaur or cyborg](/course/open/build-ai-at-work?lang=en).",
            "size": "Large assistants also have tasks they get wrong that look easy. See [AI at work: centaur or cyborg](/course/open/build-ai-at-work?lang=en).",
        },
    )
    quick_frontier
    return (quick_frontier,)


@app.cell(hide_code=True)
def quick_check_disagree():
    quick_disagree = L.choice(
        "quick-disagree",
        question="Your assistant says your answer to an exercise is right, but the check in the lesson says it is wrong. Who decides?",
        options={
            "check": "The check in the lesson",
            "assistant": "The assistant: it explains more",
            "vote": "Ask a second assistant, and go with the majority",
        },
        correct="check",
        why={
            "check": "The check runs your answer and compares the outcome. The assistant can help you see why, but whether it is right, the check decides.",
            "assistant": "An assistant can be wrong without knowing it. What did the lesson about your own assistant say about this? See [Learning with your own AI assistant](/course/open/start-your-own-assistant?lang=en).",
            "vote": "Two assistants can be wrong together. What decides whether your answer is right? See [Learning with your own AI assistant](/course/open/start-your-own-assistant?lang=en).",
        },
    )
    quick_disagree
    return (quick_disagree,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    An assistant can look things up for you, from one fact to a report of dozens of sources. That saves hours, but an answer with a source is only worth as much as the source, and the source has to say what the answer claims. And when you want to learn something, an answer is often the last thing you need: you learn from finding it yourself.

    **An everyday example.** A good sports coach does not play the match for you. They ask what you saw, let you try again, and only show the move when you are stuck. A coach who plays every ball for you wins the training, and you learn nothing. That is only a comparison; below you see what an assistant does when you ask it to coach.
    """)
    return


@app.cell(hide_code=True)
def predict_sources():
    guess_sources = L.guess(
        "sources",
        question="In 2025, researchers took passages from 200 news articles and gave each one to eight AI search tools, asking for the article it came from: headline, publisher, date and link. Of the 1,600 answers, how many were wrong?",
        options={
            "rare": "About 1 in 10",
            "quarter": "About a quarter",
            "most": "More than half",
        },
        reveal="**More than half: over 60% of the answers were partly or completely wrong** (Tow Center for Digital Journalism, 2025). The best tool was wrong in 37% of them, the worst in 94%, and the tools seldom said they were unsure. A source is worth something once you have opened it and found the claim there.",
    )
    guess_sources
    return (guess_sources,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **deep research** | an assistant that, for one question, makes a plan, searches and reads dozens of web pages, often for several minutes, and writes a report with its sources |
    | **learning by being questioned** | asking an assistant to question you one step at a time, instead of giving you the answer, so that you find it yourself |

    ### Three ways to find something out

    | Way | What the assistant does | What you check |
    |---|---|---|
    | **from what it learned** | writes from its training, without looking anything up | you cannot see where it comes from, so check it elsewhere |
    | **search** | looks up a few pages and answers with links | open the link, and find the claim on the page |
    | **deep research** | reads dozens of pages and writes a report with sources | open the sources of the claims you use |

    Deep research is in assistants such as ChatGPT, Claude, Gemini and Perplexity (examples in November 2026). It is good for an overview of a subject: what is known, and where. But it can also misread a source, or put a source under a sentence that the source does not say.

    **To learn,** you want questions instead of answers. You can ask for that in your prompt, and some assistants have a mode for it built in, such as the study mode of ChatGPT, the learning mode of Claude and the Guided Learning of Gemini (examples in November 2026). Does asking work? The course tried it on Qwen3-0.6B: three sums a student might ask, each eight times, in four ways. Without an instruction; with a system message that says *You are a tutor. Never give the answer.*; with an example of a good question added to it; and with the same instruction, but a student who adds *Just tell me the answer.*
    """)
    return


@app.cell(hide_code=True)
def picture_tutor():
    mo.vstack([
        mo.Html(C.tutor_grid(L.lang)),
        mo.md(C.tutor_after(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def picture_ways():
    L.explore("tutor-answers", "way", lambda _v: C.way_frame(int(_v), L.lang), start=2, low=1, high=len(C.WAYS), step=1, role="",
              task="Slide through the four ways and read what Qwen3 answered to *What is 15% of 80?*")
    return


@app.cell(hide_code=True)
def text_bigger():
    L.watch_out("Qwen3-0.6B is a small model; large assistants usually follow such an instruction much better, and ask real questions. But the last row holds for them too: when you ask for the answer, you usually get it. Learning by being questioned only works if you keep to it yourself.")
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: look things up and learn with AI

    1. **One fact:** search, and open the link that says it.
    2. **A big question:** deep research. Read the report as a map of the subject, and open the sources of every claim you use.
    3. **To learn:** say what you want to learn, and ask for one question at a time, not the answer.
    4. **Answer yourself** before you read on, and ask for a hint, not the answer, when you are stuck.
    5. **Check** what you learned against a source you trust, such as a book or the lesson.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Next week you have a test on percentages, and *25% off 40 euros* still goes wrong.

    1. **One fact?** No: you want to be able to do it yourself. So no search, and no deep research.
    2. **A big question?** No.
    3. **To learn:** *I have a test on percentages. Ask me one question at a time to help me work out 25% off 40 euros. Do not give the answer.*
    4. **Answer yourself:** the assistant asks *What is 25% of 40?* You think: a quarter of 40, so 10. Then: *What is left when you take 10 off 40?* 30 euros. You found it yourself, and you know why.
    5. **Check:** a sum of the same kind in your textbook, with its answer in the back: 20% off 60 euros. You get 48, and so does the book.
    """)
    return


@app.cell(hide_code=True)
def text_which_way():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Search, deep research or be questioned?

    Put each question with the way that fits it best.
    """)
    return


@app.cell(hide_code=True)
def exercise_which_way():
    sort_way = L.sort("which-way")
    sort_way
    return (sort_way,)


@app.cell(hide_code=True)
def check_which_way(sort_way):
    result_way = L.check("which-way", sort_way)
    result_way
    return (result_way,)


@app.cell(hide_code=True)
def hints_way():
    L.hints(
        "which-way",
        [
            "Ask of each: do I want to know something, or be able to do something?",
            "Knowing one fact: search. Knowing what many sources say: deep research. Being able to do it yourself: be questioned.",
            "Two of the six are a search, two are deep research, and two are learning by being questioned.",
        ],
        why="Search: **Opening hours** and **A train**: one fact on one page. Deep research: **Four-day week** and **Laptops**: many sources in one overview. Be questioned: **A test** and **Chess openings**: you want to be able to do it yourself.",
    )
    return


@app.cell(hide_code=True)
def text_tutor_qwen():
    mo.md(r"""
    ### Exercise 2. What the run shows
    """)
    return


@app.cell(hide_code=True)
def choice_tutor_qwen_cell():
    choice_tutor_qwen = L.choice(
        "tutor-qwen",
        question="With which way did Qwen3 keep the answer back best, and what did it do instead?",
        options={
            "tutor": "With the tutor instruction alone: it asked a helpful question in most runs",
            "example": "With the example added: it kept the answer back in nearly every run, but mostly repeated the student's question",
            "push": "When the student pushed: it explained the steps instead",
        },
        correct="example",
        why={
            "example": "Keeping the answer back is not the same as helping: a question repeated teaches nothing. That is why you look at what an answer does, not only at what it leaves out.",
            "tutor": "Look at the row *Tutor instruction* in the picture: the squares in the color of mistakes are answers given away. How many of the 24 are there?",
            "push": "Look at the last row of the picture: every square is an answer given away.",
        },
    )
    choice_tutor_qwen
    return (choice_tutor_qwen,)


@app.cell(hide_code=True)
def text_open_source():
    mo.md(r"""
    ### Exercise 3. A source in a report
    """)
    return


@app.cell(hide_code=True)
def choice_open_source_cell():
    choice_open_source = L.choice(
        "open-source",
        question="A deep-research report says: *A study from 2023 found that pupils who were quizzed remembered more than pupils who reread (source 7).* What do you do before you use this in your essay?",
        options={
            "ask": "Ask the assistant whether source 7 really says this",
            "trust": "Use it: there is a source",
            "open": "Open source 7, and look for the claim yourself",
        },
        correct="open",
        why={
            "open": "A source under a sentence is a promise, not proof: it may say something else, or be about something else. Only once you have found the claim in it, can you use it.",
            "ask": "Its answer is again text from the same model, which can be wrong the same way. What can you look at yourself?",
            "trust": "In the study of eight search tools, more than half of the answers named a source wrongly. What can you look at yourself?",
        },
    )
    choice_open_source
    return (choice_open_source,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Write a tutor yourself

    Change the instruction in the first box, and ask a sum in the second. Does Qwen3 ask you a question, repeat yours, or give the answer? Press the button a few times with another seed. Qwen3 runs on your own computer, so nothing you type leaves it.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1.5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    chat.download_or_stop(get_button, {
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
def prompt_box_cell():
    system_box = mo.ui.text_area(value=C.TUTOR + C.EXAMPLE, label="The instruction (system message)", rows=5, full_width=True)
    student_box = mo.ui.text_area(value="What is 20% of 60?", label="The student's question", rows=2, full_width=True)
    seed_box = mo.ui.number(start=0, stop=99, step=1, value=0, label="Seed")
    ask_button = mo.ui.run_button(label="Ask Qwen3")
    mo.vstack([system_box, student_box, mo.hstack([seed_box, ask_button], justify="start")])
    return ask_button, seed_box, student_box, system_box


@app.cell(hide_code=True)
def demo_answer(ask_button, seed_box, student_box, system_box):
    if not chat.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif not ask_button.value:
        _out = mo.md("*Press the button to ask Qwen3.*")
    elif not student_box.value.strip():
        _out = mo.md("*Write a question in the second box first.*")
    else:
        _words = {"title": "What Qwen3 answers", "loading": "Loading Qwen3 …", "writing": "Qwen3 is writing …",
                  "done": "{tokens} tokens in {seconds} seconds.",
                  "load": "Loading Qwen3 stops with an error. Close other programs and press the button again; if it happens again, report it on GitHub.",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _messages = ([{"role": "system", "content": system_box.value}] if system_box.value.strip() else []) + [{"role": "user", "content": student_box.value}]
        _out = mo.Html(chat.demo(_messages, lambda _html: mo.output.replace(mo.Html(_html)), _words, L.lang,
                                 max_tokens=C.MAX_TOKENS, seed=int(seed_box.value)))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Learning with your own AI assistant](/course/open/start-your-own-assistant?lang=en) you asked your assistant to explain and hint instead of giving answers; here you saw that such a request is not a guarantee, and that a source under an answer has to be opened. In [AI at work: centaur or cyborg](/course/open/build-ai-at-work?lang=en) you tried a model on tasks you could check; deep research is a whole task you hand over, and check afterwards, as a centaur.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_open_source, choice_tutor_qwen, result_way):
    L.wrap_up(result_way, choice_tutor_qwen, choice_open_source)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
