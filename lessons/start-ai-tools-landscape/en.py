# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson start-ai-tools-landscape. Route: course/route/lessen/bouwen.yaml. Checks and the made-up chat: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("start-ai-tools-landscape", "en")
    C = L.checks   # the made-up chat for the picture of a context window


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Using AI tools with your eyes open](/course/open/start-using-ai-tools?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bakery():
    quick_bakery = L.choice(
        "quick-bakery",
        question="An assistant writes that the small bakery on your street opened in 1987. Why do you check that before it goes into the school paper?",
        options={
            "likely": "It writes a likely year, and for a rare fact likely is often not true.",
            "never": "Assistants never get years right.",
            "purpose": "Assistants make up years on purpose, to sound sure.",
        },
        correct="likely",
        why={
            "likely": "The model writes a likely next word each time. Few texts mention your bakery, so a likely year can easily be wrong, and it sounds just as sure.",
            "never": "For things many texts mention, an assistant usually gets years right. What makes this year different? See [Using AI tools with your eyes open](/course/open/start-using-ai-tools?lang=en).",
            "purpose": "It has no purpose of its own: it picks a likely next word, word by word. Why can that go wrong for this bakery? See [Using AI tools with your eyes open](/course/open/start-using-ai-tools?lang=en).",
        },
    )
    quick_bakery
    return (quick_bakery,)


@app.cell(hide_code=True)
def quick_check_no_check():
    quick_no_check = L.choice(
        "quick-no-check",
        question="Which part of an assistant's answer can you use without checking it?",
        options={
            "quote": "A quote from a famous person",
            "title": "An idea for a title that you choose yourself",
            "date": "The date of an event",
        },
        correct="title",
        why={
            "title": "An idea you choose from yourself is not a fact, so there is nothing to check.",
            "quote": "A quote is a fact: did that person really say it? Step 2 of the recipe in [Using AI tools with your eyes open](/course/open/start-using-ai-tools?lang=en) picks out the facts.",
            "date": "A date is a fact, and exact numbers are where assistants slip. Step 2 of the recipe in [Using AI tools with your eyes open](/course/open/start-using-ai-tools?lang=en) picks out the facts.",
        },
    )
    quick_no_check
    return (quick_no_check,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    There are hundreds of AI tools, and new ones come out every month. The one you pick decides what it can do for you, how much it can read at once, and who else gets what you give it. Names change fast, but the kinds of tools stay the same, and so do the questions you ask about each.

    **An everyday example.** You can eat at a restaurant, order a meal that comes with your groceries, or cook at home. The restaurant has the biggest kitchen, but the staff see what you order. At home your kitchen is smaller, but nobody sees what you cook. That is only a comparison; below you see the three kinds of AI tools it stands for.
    """)
    return


@app.cell(hide_code=True)
def predict_where():
    guess_where = L.guess(
        "where-it-goes",
        question="You paste a letter from your doctor into a free chat assistant on the web, and ask for a summary. Where is the letter after that?",
        options={
            "only": "Only on your own computer",
            "theirs": "Also on the computers of the company behind the assistant",
            "gone": "Nowhere: it is deleted as soon as you have the answer",
        },
        reveal="**Also on the company's computers.** The model does not run on your computer: your text goes over the internet to the company that runs it, the provider. There your chat is usually stored. Depending on the provider and your settings, people at the company may read parts of it, for example to check for misuse, and it may be used to train later models. So your letter is now with a company whose terms you probably never read.",
    )
    guess_where
    return (guess_where,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **context window** | how much text a model can look at, at most, when it writes the next word: your messages, its answers and everything added to the chat |
    | **open weights** | a model is published, so anyone can download it and run it on computers of their own; small ones even on a laptop |
    | **privacy hygiene** | the habit of checking what you give a tool before you paste or upload it, and leaving out what is not for everyone |

    ### Three kinds of AI tools

    | Kind | Examples in November 2026 | Where your text goes |
    |---|---|---|
    | a **chat assistant** in a browser or an app | ChatGPT, Claude, Gemini, Microsoft Copilot, Le Chat | to the provider's computers |
    | **AI inside a program** you already use | writing help in an email app, suggestions in a code editor, a photo app that removes things from a picture | usually to the maker's computers, or those of an AI company it works with; a phone does some of it on the phone itself |
    | a **model you run yourself** | open-weights models such as Qwen, Mistral, Gemma and gpt-oss, with a program that runs them | nowhere: it stays on your computer |

    The names in the middle column are the well-known ones at the time of writing; next year some will be new. The right column is the question that stays.

    ### What an assistant can do besides writing

    Many assistants can **search the web**, **read files** you give them (a document, a spreadsheet, a photo), **run code** to work out a sum or draw a chart, and **remember** things from earlier chats. Some make pictures or talk out loud. A later lesson shows assistants that take actions for you, step by step.

    All of that gives the model more text to look at: the search results, the file, the outcome of the code. The language model still writes the answer word by word. And everything it looks at has to fit in its context window.
    """)
    return


@app.cell(hide_code=True)
def text_window():
    mo.md(r"""
    ### The context window

    For every next word, the model looks at the chat so far. But it can look at only so much at once: its context window. That is counted in pieces of words (a later lesson calls them tokens and shows how text is cut up). When a chat gets longer than the window, the start no longer fits. Depending on the tool, it drops the oldest messages, as in the picture below, makes a short summary of them, or asks you to start a new chat. Either way, the model no longer sees the start word for word.

    Real assistants hold far more than the picture: from about ten thousand pieces to a million or more, depending on the tool and on what you pay. A long chat with long texts pasted into it can still be too much. And even inside the window, a detail in a very long text is missed more often than in a short one.
    """)
    return


@app.cell(hide_code=True)
def picture_window():
    window_story = L.explore(
        "window-story", "messages in the chat", lambda n: C.window_frame(n, L.lang), start=8, low=1, high=8, step=1, role="",
        task="An assistant writes a story with you. Slide back from 8 messages: from which message on does the model no longer see that Noor is afraid of water?")
    mo.vstack([
        window_story,
        L.watch_out("A context window is not a window on your screen, like the chat window: it is how much of the chat the model can look at at once."),
        mo.md(r"""
        Each row is a message, with its length in pieces of words; the lengths and the window of 1,000 pieces are made up, so that it fits in the picture. The **bracket** marks the newest messages that fit in the context window together; the grey ones fell out. From message 6 on, the model no longer sees message 1, and in chapter 4 it lets Noor dive into the sea.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_open_weights():
    mo.md(r"""
    ### Open weights

    The biggest assistants run on rows of special computers in a data center: you only ever get their answers, never the model itself. Some companies publish their models instead. A model with **open weights** can be downloaded by anyone (a later lesson shows what the numbers inside it are) and run on computers of your own, without the internet; small ones even run on a laptop. Then nothing you type leaves your computer. The catch: a model small enough for a laptop knows less and makes more mistakes than the biggest assistants. Later in the course you run one yourself.

    Open weights is not the same as open source: the company often keeps the texts it trained on and its own code to itself, and its license can still set conditions.
    """)
    return


@app.cell(hide_code=True)
def picture_where():
    mo.vstack([
        draw.flow(
            [("Your computer", ["you type or paste"]), ("The provider's computers", ["the model writes the answer", "your chat is stored"])],
            ["your text, over\nthe internet"], ["the answer"], title="A chat assistant on the web"),
        draw.flow(
            [("Your computer", ["you type or paste", "an open-weights model writes the answer", "nothing leaves"])],
            [], [], title="A model you run yourself"),
        mo.md(r"""
        With a chat assistant, and with most AI inside other programs, your text travels to the provider; with a model on your own computer, it stays where it is.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Deleting a chat in the app does not always delete it at once on the provider's computers: many keep it a while longer, for example for safety checks. The moment that counts is the moment you paste."),
        mo.md(r"""
        ### Recipe: think before you paste

        1. **Will it leave your computer?** With a chat assistant and with most AI inside programs: yes. With a model you run yourself: no, and you can stop here.
        2. **What in it is not for everyone?** Passwords and access codes; details about yourself, such as your address, bank account or health; details about other people; and what belongs to your school or work.
        3. **Leave it out or replace it.** Write *[name]* instead of a name, use made-up numbers, write *a colleague*. A password never goes into a chat at all.
        4. **Material from school or work** goes only into a tool that your school or employer has approved, under its rules.
        5. **Look at the settings** once: does the provider use your chats to train later models? Many tools let you turn that off.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Your energy bill is far too high, and you want help writing a complaint. The bill has your name, address, account number and bank account number on it, and your usage per month with the amounts. You want to use a free chat assistant on the web.

    1. **Will it leave your computer?** Yes: it is a chat assistant on the web. Go on.
    2. **What in it is not for everyone?** Your name, address, account number and bank account number. The usage and the amounts are not about you as a person, and the letter needs them.
    3. **Leave it out or replace it.** You type the usage and the amounts, and write *[name]* and *[account number]* where those go. The assistant writes the letter; you fill in your own details afterwards, in your own text editor.
    4. **Material from school or work?** No, it is your own bill. Step 4 does not apply.
    5. **Look at the settings.** You check whether your chats are used for training, and turn it off.

    And the amounts in the letter that comes back? Those are facts the complaint depends on: you check them against your bill, with the recipe of [Using AI tools with your eyes open](/course/open/start-using-ai-tools?lang=en).
    """)
    return


@app.cell(hide_code=True)
def text_exercise_paste():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Before you paste

        You want help from a free chat assistant on the web with each of the six things below. Step 1 of the recipe is done: all six would leave your computer. Now do steps 2 to 4.
        """),
        L.task("Put each of the six in the right group."),
    ])
    return


@app.cell(hide_code=True)
def exercise_paste():
    sort_paste = L.sort("before-you-paste")
    sort_paste
    return (sort_paste,)


@app.cell(hide_code=True)
def check_before_you_paste(sort_paste):
    result_paste = L.check("before-you-paste", sort_paste)
    result_paste
    return (result_paste,)


@app.cell(hide_code=True)
def hints_paste():
    L.hints(
        "before-you-paste",
        [
            "Step 2 of the recipe: what in it is not for everyone? Think of passwords, details about people, and what belongs to your work.",
            "Something everyone may read already, such as a recipe from a book or a news article online, can go in as it is.",
            "Ask of each: who could be harmed if a stranger read it? Three are like that.",
        ],
        why="Paste as it is: **Pancakes**, **News** and **Wikipedia**. Not as it is: **Password** (never in a chat), **Letter** (leave out the name and the other details that point to your neighbor) and **Customers** (it stays at work, or goes into a tool your employer approved).",
    )
    return


@app.cell(hide_code=True)
def text_which_tool():
    mo.md(r"""
    ### Exercise 2. Which kind of tool?
    """)
    return


@app.cell(hide_code=True)
def choice_which_tool_cell():
    choice_which_tool = L.choice(
        "which-tool",
        question="A hospital wants an AI model to sum up patient files, and the files may not leave the hospital's own computers. Which kind of tool fits?",
        options={
            "own": "An open-weights model that runs on the hospital's own computers",
            "chat": "A chat assistant on the web, with training on chats turned off",
            "inside": "The AI inside the email app the hospital already uses",
        },
        correct="own",
        why={
            "own": "Only a model that runs on the hospital's own computers keeps the files there. That it may know less than the biggest assistants is the price.",
            "chat": "Turning off training does not keep the files in the hospital: they still travel to the provider's computers. Look at the right column of the table with the three kinds.",
            "inside": "AI inside a program usually sends the text to the maker's computers too, or to an AI company it works with. Look at the right column of the table with the three kinds.",
        },
    )
    choice_which_tool
    return (choice_which_tool,)


@app.cell(hide_code=True)
def text_long_chat():
    mo.md(r"""
    ### Exercise 3. A long chat
    """)
    return


@app.cell(hide_code=True)
def choice_long_chat_cell():
    choice_long_chat = L.choice(
        "long-chat",
        question="You plan a trip in one long chat. In your first message you wrote that you don't eat meat. Since then you have pasted three long travel guides into the chat. Now the assistant suggests a steakhouse. What is the most likely reason?",
        options={
            "latest": "An assistant only reads your latest message, never the ones before it.",
            "window": "The travel guides pushed your first message out of its context window.",
            "mind": "It decided that you changed your mind, because you did not say it again.",
        },
        correct="window",
        why={
            "window": "The long travel guides filled the context window, and your first message fell out. Repeat what matters, or start a new chat with it at the top.",
            "latest": "It reads all of the chat that fits in its context window, not only the last message. What could make your first message no longer fit?",
            "mind": "If your first message were still in its context window, the model would see it at every next word, and nothing in the chat says you changed your mind. What could make the first message disappear?",
        },
    )
    choice_long_chat
    return (choice_long_chat,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Using AI tools with your eyes open](/course/open/start-using-ai-tools?lang=en) you learned to check what an assistant says. Now you also know what you give it: which kind of tool sends your text where, how much of a chat the model sees at once, and what you leave out before you paste. Later in this module you learn to ask an assistant well, and later in the course you run an open-weights model on your own computer and see a context window from the inside.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_long_chat, choice_which_tool, result_paste):
    L.wrap_up(result_paste, choice_which_tool, choice_long_chat)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
