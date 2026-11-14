# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson start-prompt-basics. Route: course/route/lessen/bouwen.yaml. Checks, the questions and the course's run: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("start-prompt-basics", "en")
    C = L.checks   # the three questions, the seven ways of asking them, and the course's run of Qwen3


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Which AI tool for what](/course/open/start-ai-tools-landscape?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_window():
    quick_window = L.choice(
        "quick-window",
        question="You have chatted with an assistant for a long time, and now it forgets something you told it at the start. What is the likely reason?",
        options={
            "last": "It only ever reads your last message",
            "window": "The start of the chat no longer fits in its context window",
            "stored": "It stores the chat, but never reads it back",
        },
        correct="window",
        why={
            "last": "It reads much more than your last message: everything that fits in its context window. See [Which AI tool for what](/course/open/start-ai-tools-landscape?lang=en).",
            "window": "The model sees only what fits in its context window; in a long chat the oldest messages fall out.",
            "stored": "What fits in its context window it reads every time it answers. See [Which AI tool for what](/course/open/start-ai-tools-landscape?lang=en).",
        },
    )
    quick_window
    return (quick_window,)


@app.cell(hide_code=True)
def quick_check_paste():
    quick_paste = L.choice(
        "quick-paste",
        question="You want a chat assistant on the web to sum up a complaint from a customer. What do you do first?",
        options={
            "nothing": "Nothing: paste it as it is",
            "summary": "Write the summary yourself, so the assistant never sees it",
            "leave-out": "Leave out or replace the customer's name and contact details",
        },
        correct="leave-out",
        why={
            "nothing": "What you paste goes to the provider's computers. Is there anything in it about someone else? See [Which AI tool for what](/course/open/start-ai-tools-landscape?lang=en).",
            "summary": "You can let it help: just not with details it does not need. See [Which AI tool for what](/course/open/start-ai-tools-landscape?lang=en).",
            "leave-out": "The summary does not need who wrote it; without those details, the complaint can go into the chat.",
        },
    )
    quick_paste
    return (quick_paste,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The same model, the same question, and still a right or a wrong answer: it depends on how you ask. A model writes its answer word by word, and every word it writes depends on everything before it, your words included. So what you put in, and in which form, changes what comes out. This lesson shows how much, with one small chat model and three school sums, each asked seven ways, eight times each.
    """)
    return


@app.cell(hide_code=True)
def predict_number():
    guess_number = L.guess(
        "only-number",
        question=f"The course asked Qwen3-0.6B, a small chat model, this question {C.SEEDS} times: *{C.QUESTIONS['bakery'][0]}* And {C.SEEDS} times the same, with *Reply with only the final number, nothing else.* added. Which way gave the right answer, {C.QUESTIONS['bakery'][1]}, more often?",
        options={
            "plain": "The question as it is",
            "same": "Both about as often",
            "number": "With only the number",
        },
        reveal=C.number_reveal(L.lang),
    )
    guess_number
    return (guess_number,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    A **prompt** is everything a model gets before it writes. In a chat, that is not only your last message but all messages so far, and each message has a **role**: `system` for the instructions of whoever runs the assistant, `user` for you, and `assistant` for the model. A good prompt has up to four parts: an *instruction*, what the model should do; *context*, what it needs to know, such as the text to work on; *examples* of what you want; and the *form* of the answer. Text you paste in goes between **delimiters**, such as `<review>` and `</review>`, so the model can tell your instruction from the text.

    Three ways of asking have names. Asking with no examples is **zero-shot**; with a few examples in the prompt, **few-shot**. Telling the model who to be, such as *You are a patient math teacher*, is **role prompting**. And asking it to write its steps before the answer is **chain of thought**.

    ### New words and symbols

    | Word | What it means | Example |
    |---|---|---|
    | **prompt** | everything a model gets before it writes; in a chat, all messages so far | |
    | **message role** | who a message comes from: `system` (whoever runs the assistant), `user` (you) or `assistant` (the model) | `{"role": "user", "content": "Hi"}` |
    | **delimiter** | a mark that sets a part of a prompt apart, such as a tag before and after a text | `<review>` … `</review>` |
    | **zero-shot prompting** | asking without examples | *Translate this into Dutch: …* |
    | **few-shot prompting** | asking with a few examples of questions and answers in the prompt | *English: Good morning. Dutch: Goedemorgen.* |
    | **role prompting** | telling the model who to be | *You are a patient math teacher.* |
    | **chain of thought** | the steps a model writes before its answer, and asking it for them | *Work it out step by step.* |
    """)
    return


@app.cell(hide_code=True)
def watch_out_role():
    L.watch_out("*Role* means two things in this lesson. The *role of a message* says who wrote it: `system`, `user` or `assistant`. *Role prompting* tells the model who to be. The two can meet: a role prompt such as *You are a patient math teacher* often goes into the message with the role `system`.")
    return


@app.cell(hide_code=True)
def picture_parts():
    mo.vstack([
        draw.marked_lines([("A prompt with all four parts", [
            ("Sort the email between the <email> tags as a complaint, a question or praise.", "1"),
            ("<email>", "2"),
            ("We waited forty minutes for our pancakes, and they were cold.", "2"),
            ("</email>", "2"),
            ("Example: \"Do you have a table for six on Friday?\" is a question.", "3"),
            ("Reply with one word: complaint, question or praise.", "4"),
        ])], {"1": "the instruction", "2": "the context, between delimiters", "3": "an example", "4": "the form of the answer"},
            title="Four parts of a prompt"),
        mo.md(r"""
        Not every prompt needs all four. Here one word is the right form: there is nothing to work out, only a choice. For a sum it is different, as the seven ways below show.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_ways():
    mo.md(rf"""
    **One question, seven ways.** The course asked Qwen3-0.6B each of three sums in the seven ways below, {C.SEEDS} times each. A model picks each next word with a little chance in it, so the same prompt can get another answer; a *seed* fixes that chance, so a run can be repeated exactly, and {C.SEEDS} seeds show what is luck and what is a pattern. The length of an answer is counted in *tokens*, the pieces of words that [Which AI tool for what](/course/open/start-ai-tools-landscape?lang=en) counted the context window in. Qwen3-0.6B is small enough for a laptop, and it writes English best, so the prompts are in English. The lesson reads the last number in each answer. Slide through the ways for the bakery question.
    """)
    return


@app.cell(hide_code=True)
def ways_cell():
    L.explore("ways", "way", lambda _v: C.way_frame(int(_v), "bakery", L.lang), start=C.WAYS.index("number") + 1, low=1,
              high=len(C.WAYS), step=1, role="", task="Find the three ways that are right least often. What do their answers have in common?")
    return


@app.cell(hide_code=True)
def text_ways_after():
    mo.md(C.table(L.lang) + "\n\n" + C.ways_after(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To ask a model well:

    1. Say what you want, as an instruction.
    2. Give the context it needs, and set a pasted text apart with delimiters.
    3. If the answer has a fixed form, show it with one or two examples, with their steps if there is something to work out.
    4. If there is something to work out, let the model write its steps first, and the answer last.
    5. Ask for the form you need at the end, such as *Answer: …* on the last line, not instead of the steps.
    6. Check the answer: a well-asked question still gets a wrong answer sometimes.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A café wants a model to sort the emails it gets as a complaint, a question or praise. A first try: *What is this email?* and the email under it. The model has to guess what you want; it may sum the email up, or answer it. The recipe, step by step:

    1. The instruction: *Sort the email as a complaint, a question or praise.*
    2. The context: the email, between `<email>` and `</email>`, so a sentence in the email such as *Please answer quickly* is not read as an instruction to the model.
    3. One example of each kind, so the model sees where the line lies: *"The pancakes were lovely" is praise.*
    4. Steps? There is nothing to calculate, so a short answer is fine.
    5. The form: *Reply with one word: complaint, question or praise.* Now another program can read the answer.
    6. Check: try it on emails whose kind you know, such as ten from last week, before you trust it with the rest.

    For the bakery sum, step 4 goes the other way: there the answer has to be worked out, and the steps are where that happens.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_parts():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The parts of a prompt

        Someone asks a model to translate a message from their landlord. Sort the six lines of their prompt: which part of a prompt is each?
        """),
        L.task("Put each of the six lines in its group."),
    ])
    return


@app.cell(hide_code=True)
def exercise_parts():
    sort_parts = L.sort("parts")
    sort_parts
    return (sort_parts,)


@app.cell(hide_code=True)
def check_parts(sort_parts):
    result_parts = L.check("parts", sort_parts)
    result_parts
    return (result_parts,)


@app.cell(hide_code=True)
def hints_parts():
    L.hints(
        "parts",
        [
            "Ask of each line: does it say what to do, what the model needs to know, what a good answer looks like, or how the answer must be written?",
            "Two lines tell the model something it needs to know, and one shows a finished answer. Find those three first.",
            "The instruction is the line that says to translate; the two lines that limit the answer are the form.",
        ],
        why="*Translate the message between the tags into Dutch* is the instruction. The message between the tags and *The message is from my landlord* are the context. *English: Good morning! Dutch: Goedemorgen!* is an example. *Reply with the translation only* and *Use at most two sentences* say how the answer must be written: the form.",
    )
    return


@app.cell(hide_code=True)
def text_sum():
    mo.md(r"""
    ### Exercise 2. A sum for a small model
    """)
    return


@app.cell(hide_code=True)
def choice_sum_cell():
    choice_sum = L.choice(
        "sum-prompt",
        question="You want Qwen3-0.6B to work out how many minutes are in three and a half hours, and another program reads the number. Which prompt gets the right number most often, going by the table of the seven ways?",
        options={
            "professor": "*You are a math professor. Reply with only the number.*",
            "steps": "*Work it out step by step. Then give the answer on the last line, as: Answer: &lt;number&gt;*",
            "number": "*Reply with only the number, nothing else.*",
        },
        correct="steps",
        why={
            "professor": "A role changes how it writes, not how well it calculates, and *only the number* leaves it little room to work the sum out. Look at the role and the only-the-number ways in the table again.",
            "steps": "The steps give the model room to work it out, and *Answer:* on the last line gives the other program a fixed place to read the number.",
            "number": "With only the number, the model writes little before its answer, and in the table such short answers were wrong more often. Look at the only-the-number way again.",
        },
    )
    choice_sum
    return (choice_sum,)


@app.cell(hide_code=True)
def text_role_of():
    mo.md(r"""
    ### Exercise 3. Who wrote it?
    """)
    return


@app.cell(hide_code=True)
def choice_role_cell():
    choice_role = L.choice(
        "message-role",
        question="The makers of a cooking app put *Always answer in Dutch, and only about cooking.* before every chat of their users. Which role does that message have?",
        options={
            "user": "`user`",
            "assistant": "`assistant`",
            "system": "`system`",
        },
        correct="system",
        why={
            "user": "`user` is the person who chats. This message comes from the makers of the app, before the chat starts. Look at the table of new words again.",
            "assistant": "`assistant` is what the model wrote. This message is written for the model, by the makers of the app. Look at the table of new words again.",
            "system": "Instructions of whoever runs the assistant go into the message with the role `system`, at the start of every chat.",
        },
    )
    choice_role
    return (choice_role,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Ask Qwen3 yourself

    Write your own prompt for Qwen3-0.6B, and see what it answers: one of the three sums in your own way, or anything else. The answer is sampled with seed 0, as in the course's first try of each way. Qwen3 runs on your own computer, so nothing you type leaves it.
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
def prompt_box_cell():
    prompt_box = mo.ui.text_area(value=C.messages("steps", C.QUESTIONS["bakery"][0])[0]["content"],
                                 label="Your prompt", rows=5, full_width=True)
    ask_button = mo.ui.run_button(label="Ask Qwen3")
    mo.vstack([prompt_box, ask_button])
    return ask_button, prompt_box


@app.cell(hide_code=True)
def demo_answer(ask_button, prompt_box):
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif not ask_button.value:
        _out = mo.md("*Press the button to ask Qwen3.*")
    elif not prompt_box.value.strip():
        _out = mo.md("*Write a prompt in the box first.*")
    else:
        _words = {"title": "What Qwen3 answers", "loading": "Loading Qwen3 …", "writing": "Qwen3 is writing …",
                  "done": "{tokens} tokens in {seconds} seconds; the last number in it: {read}", "nothing": "none",
                  "load": "Loading Qwen3 stops with an error. Close other programs and press the button again; if it happens again, report it on GitHub.",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.ask_live(prompt_box.value, lambda _html: mo.output.replace(mo.Html(_html)), _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Which AI tool for what](/course/open/start-ai-tools-landscape?lang=en) you saw that a model reads everything in its context window; a prompt is what you put there, and its form changes the answer. The next lesson, [Prompts that check themselves](/course/open/start-prompt-chains?lang=en), splits a big task into small prompts in a row, and lets a model check its own answer. In the domain *Large language models* you will see why a chat model has message roles at all, and how chain of thought can be trained into a model instead of asked for.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_role, choice_sum, result_parts):
    L.wrap_up(result_parts, choice_sum, choice_role)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
