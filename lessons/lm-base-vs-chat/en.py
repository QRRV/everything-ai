# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson lm-base-vs-chat. Route: course/route/lessen/taal.yaml. Checks: checks.py.
# termen:allow instruct model, instruct models

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-base-vs-chat", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_writes():
    quick_writes = L.choice(
        "quick-writes",
        question="How does a language model write a text?",
        options={
            "lookup": "It looks up a whole answer that fits the text so far",
            "right": "It picks a next token from its probabilities, adds it, and does it again",
            "once": "It works out all the tokens at once, and then shows them",
        },
        correct="right",
        why={
            "lookup": "A language model only gives probabilities for the next token; a text comes from doing that again and again. See [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en).",
            "once": "It gives the probabilities of one next token at a time, and each new token depends on the ones before it. See [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en).",
            "right": "Each new token goes into the text so far, and the model gives the probabilities of the next one.",
        },
    )
    quick_writes
    return (quick_writes,)


@app.cell(hide_code=True)
def quick_check_labels():
    quick_labels = L.choice(
        "quick-labels",
        question="A language model learns by next-token prediction on a lot of text. Where do the labels come from?",
        options={
            "people": "People write the right answer for each sentence",
            "model": "The model makes them up while it learns",
            "right": "From the text itself, one label per next token",
        },
        correct="right",
        why={
            "people": "Nobody adds labels: the text gives its own labels. See [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en).",
            "model": "The labels are there before training starts: the tokens of the text. See [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en).",
            "right": "The token that really comes next is the label, so a model can learn from any text without anyone adding labels.",
        },
    )
    quick_labels
    return (quick_labels,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You type a question into a chat program, and an answer comes back. Yet the model behind it only does what [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en) showed: it gives probabilities for the next token. A model that learned from text alone has seen a question followed by many things: another question in a list of exam questions, a reply in a forum, the rest of a story. So what makes it answer you?

    **An everyday example.** Someone who has read thousands of books knows how sentences go on. Put them behind a help desk for the first time, and they still have to learn what a good answer to a customer is: short, to the point, and about the question. They learn that on the job, from many examples.
    """)
    return


@app.cell(hide_code=True)
def predict_turn():
    guess_turn = L.guess(
        "plays-turn",
        question="The course's model learned from the plays of Shakespeare, and from nothing else. It gets the text below, and writes on until it starts an empty line. What do you think it writes?\n\n```\nUSER:\nWhat is a token?\n\nASSISTANT:\n```",
        options={
            "right": "A few lines of text in the style of the plays",
            "answer": "An answer that says a token is a piece of text",
            "nothing": "Nothing, as the plays never mention tokens",
        },
        reveal=f"**A few lines of text in the style of the plays.** With seed 3 it writes:\n\n{C.quoted(C.write_turn(C.GUESS_TEXT, seed=3))}\n\nIn the plays, a name with a colon on a line of its own, such as `ROMEO:`, is followed by what that person says, up to an empty line. So after `ASSISTANT:` the model writes a speech, as if a character called Assistant speaks: it takes its turn, but it does not answer. It never learned what a good answer is, only what text comes next in the plays. And it does write something: a language model always gives probabilities for a next token.",
    )
    guess_turn
    return (guess_turn,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **base model** is a language model that has only learned next-token prediction, on a lot of text. It continues whatever text it gets, with what tends to come next in the texts it learned from. The course's model of the plays is a base model, a very small one. It counts which character follows the last 7 characters in the plays, as the model of [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en) did with 3; when those 7 never occur, it looks at fewer. So it does not even see the question: with the same seed, it writes the same after `ASSISTANT:`, whatever was asked. A real base model sees the whole text, and still only continues it. A large one that has seen many questions with answers on the web may continue with something like an answer, but just as easily with another question, or the rest of a quiz.

        A **chat model** is a base model that was trained further on conversations: many examples in which a person asks something and an assistant answers well. It still does nothing but predict the next token; only, after a question, the most likely text is now an answer. Chat programs use chat models. Models whose name ends in -Instruct or -Chat are chat models; some texts call them instruct models.

        A conversation is made of messages, and each message has a **role** that says who it comes from: `system` for instructions from whoever runs the model, such as "Answer in Dutch."; `user` for the person who types; `assistant` for the model itself. In code, a conversation is a list of messages, each a dictionary with the keys `"role"` and `"content"`: that is how most programs hand a conversation to a chat model. Some programs add a fourth role, `tool`, for what a tool that the model used gave back.

        Yet the model still gets one text. The conversation is written out with a marker before each message, and it ends with the marker of the assistant, so that what comes next is the answer. The model writes until it gives the marker that ends its turn. Chat models differ in their markers, which they learned in training; the lesson *Chat templates* shows them. In this lesson the marker is the role in capitals with a colon, on a line of its own, as the plays write the name of a speaker, and an empty line ends a turn.

        The model itself keeps nothing from one turn to the next. For every new answer, the whole conversation goes in again as one text, with the model's earlier answers in it as `assistant` messages.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **base model** | | a language model that only learned to predict the next token, on a lot of text; it continues any text it gets | |
        | **chat model** | | a base model trained further on conversations, so that it answers as an assistant | |
        | **role** | | who a message comes from: `system`, `user` or `assistant` | `m["role"]` |
        | `system` | | instructions from whoever runs the model | `{"role": "system", "content": "Answer in Dutch."}` |
        | `user` | | the person who types | `{"role": "user", "content": "Who is Romeo?"}` |
        | `assistant` | | the model, also in its earlier answers | `{"role": "assistant", "content": "A young man of Verona."}` |
        """),
        L.watch_out("`assistant` is a role in the text, not a separate program: the same model that predicts every next token writes every `assistant` message. And a `system` message is text like the rest, not a lock: a `user` message can still lead the model somewhere else."),
        mo.md(r"""
        The picture: the conversation of the worked example below, written out as the one text the model gets. The model's own turn is in colour.
        """),
        mo.Html(C.turn_picture(C.EXAMPLE, C.write_turn(C.as_text_course(C.EXAMPLE), seed=6),
                               {"title": "What the model gets, and what it writes", "system": "system: who runs the model",
                                "user": "user: the person", "assistant": "assistant: the model", "end": "empty line: end of the turn",
                                "alt": "The conversation written out as one text, line by line: SYSTEM: and the instruction, an empty line, USER: and the question, an empty line, then ASSISTANT: and the line the model writes, in colour, up to an empty line that ends its turn. On the right, the role of each part."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To hand a conversation to a language model:

    1. Write the conversation as a list of messages: each a dictionary with the keys `"role"` and `"content"`.
    2. Write each message out: its marker on a line of its own, then the content, then an empty line.
    3. End with the marker of the assistant: what comes next is the answer.
    4. Let the model continue the text, up to the empty line that ends its turn.
    5. For the next turn, add the answer as an `assistant` message and the new question as a `user` message, and start again at step 2.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A conversation with a `system` message and one question, handed to the course's base model with seed 6. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_turn():
    _MARKER = {"system": "SYSTEM:", "user": "USER:", "assistant": "ASSISTANT:"}
    # 1. the conversation: a list of messages, each with a role and a content
    _messages = [
        {"role": "system", "content": "Answer in one line."},
        {"role": "user", "content": "Who is Romeo?"},
    ]
    # 2. each message written out: its marker on a line of its own, the content, an empty line
    _text = ""
    for _m in _messages:
        _text = _text + _MARKER[_m["role"]] + "\n" + _m["content"] + "\n\n"
    # 3. end with the marker of the assistant
    _text = _text + _MARKER["assistant"] + "\n"
    # 4. the course's base model continues the text, up to the empty line that ends its turn
    _reply = C.write_turn(_text, seed=6)
    print(_text + _reply)
    # 5. the next turn: the reply as an assistant message, the new question as a user message
    _messages.append({"role": "assistant", "content": _reply})
    _messages.append({"role": "user", "content": "And who is Juliet?"})
    print(len(_messages), _messages[2])
    return


@app.cell(hide_code=True)
def text_after_example():
    _reply = C.write_turn(C.as_text_course(C.EXAMPLE), seed=6)
    mo.md(f"""
    The text has three parts: the `system` message, the `user` message, and the marker `ASSISTANT:` at the end. The model writes "{_reply}" and stops at the empty line. That is no answer about Romeo: the model never learned to answer, and with 7 characters to go on, it does not even see the question. A chat model would have learned from many conversations that after a question about Romeo comes an answer about Romeo. For the next turn, the list has four messages: the model's own line is in it as an `assistant` message, because the model itself remembers nothing.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chat():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. What makes a chat model?
    """)
    return


@app.cell(hide_code=True)
def choice_chat_cell():
    choice_chat = L.choice(
        "base-to-chat",
        question="What turns a base model into a chat model?",
        options={
            "system": "A `system` message that tells it to answer as an assistant",
            "markers": "The markers `USER:` and `ASSISTANT:` before the messages",
            "right": "More training, on many conversations",
        },
        correct="right",
        why={
            "system": "A `system` message is text like the rest: a base model only continues it, as the course's model continued the text of the guess after `ASSISTANT:`.",
            "markers": "The text of the guess had those markers, and the course's base model still wrote a speech. Markers only help a model that learned in training what comes after them.",
            "right": "Training on many conversations in which an assistant answers well makes an answer the likely text after a question. The markers and the roles only say whose turn it is.",
        },
    )
    choice_chat
    return (choice_chat,)


@app.cell(hide_code=True)
def text_exercise_role():
    mo.md(r"""
    ### Exercise 2. Which role?
    """)
    return


@app.cell(hide_code=True)
def choice_role_cell():
    choice_role = L.choice(
        "role-of",
        question="A school puts a chat model on its website, and before every conversation it gives the model the instruction \"Only answer questions about the timetable.\" Which role does that message have?",
        options={
            "user": "`user`",
            "right": "`system`",
            "assistant": "`assistant`",
        },
        correct="right",
        why={
            "user": "`user` is the person who types the questions. The school runs the model, and its instructions are a `system` message.",
            "assistant": "`assistant` is for the model's own messages. The instruction comes from whoever runs the model.",
            "right": "The school runs the model, and instructions from whoever runs the model are `system` messages.",
        },
    )
    choice_role
    return (choice_role,)


@app.cell(hide_code=True)
def text_exercise_as_text():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The conversation as one text

        `as_text(messages)` writes a conversation out as one text, as in steps 2 and 3 of the recipe. The dictionary `MARKER`, the loop over the messages and the last line, with the marker of the assistant, are there. Replace the `...` with one line for step 2. The check calls `as_text([{"role": "user", "content": "Hello"}])`, which should give `"USER:\nHello\n\nASSISTANT:\n"`, and tries other conversations: with a `system` message, with earlier answers, and an empty one.
        """),
        L.task("Make `as_text` add each message to `text`: its marker on a line of its own, the content, and an empty line."),
    ])
    return


@app.cell(hide_code=True)
def exercise_as_text():
    box_as_text = L.code_box("as-text", lines=8)
    box_as_text
    return (box_as_text,)


@app.cell(hide_code=True)
def check_as_text(box_as_text):
    result_as_text = L.check("as-text", box_as_text.value)
    result_as_text
    return (result_as_text,)


@app.cell(hide_code=True)
def hints_as_text():
    L.hints(
        "as-text",
        [
            "Step 2 of the recipe, as in the worked example.",
            "The marker of a message is in `MARKER`, under its role: look up `m[\"role\"]`. After the marker comes `\"\\n\"`, after the content `\"\\n\\n\"`: the end of the line and an empty line.",
            "`text = text + MARKER[___] + \"\\n\" + m[___] + \"\\n\\n\"`.",
        ],
        why="For `[{\"role\": \"user\", \"content\": \"Hello\"}]` the loop runs once: `text` becomes `\"USER:\\nHello\\n\\n\"`, and the last line adds `\"ASSISTANT:\\n\"`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_next_turn():
    mo.vstack([
        mo.md(r"""
        ### Exercise 4. The next turn

        `next_turn(messages, reply, question)` gives the conversation for the next turn, as in step 5 of the recipe: the messages so far, then the model's reply, then the new question. The copy of the messages so far is there: `new` gets the additions, so that the list you got stays as it was. Replace the `...` with two lines. The check calls `next_turn([{"role": "user", "content": "Hello"}], "Good day to you.", "Who are you?")`, which should give a list of three messages, and tries other conversations.
        """),
        L.task("Make `next_turn` add the reply and the question to `new`, each as a message with the right role."),
    ])
    return


@app.cell(hide_code=True)
def exercise_next_turn():
    box_next_turn = L.code_box("next-turn", lines=7)
    box_next_turn
    return (box_next_turn,)


@app.cell(hide_code=True)
def check_next_turn(box_next_turn):
    result_next_turn = L.check("next-turn", box_next_turn.value)
    result_next_turn
    return (result_next_turn,)


@app.cell(hide_code=True)
def hints_next_turn():
    L.hints(
        "next-turn",
        [
            "Step 5 of the recipe, as in the worked example.",
            "Who wrote the reply, and who asks the question? Each becomes a dictionary with a `\"role\"` and a `\"content\"`, added to `new` with `append`.",
            "`new.append({\"role\": ___, \"content\": reply})`, and the same for the question.",
        ],
        why="The model wrote the reply, so it is an `assistant` message; the question comes from the person, so it is a `user` message. In that order: the reply came before the new question.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** A base model does what [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en) showed, and nothing more: it continues text. A chat model does the same, after more training on conversations; the roles and the markers turn a conversation into one text that it can continue. In the next lesson, *Why language models make things up*, you see what it means that even a chat model gives likely text, not checked facts.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_chat, choice_role, result_as_text, result_next_turn):
    L.wrap_up(choice_chat, choice_role, result_as_text, result_next_turn)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
