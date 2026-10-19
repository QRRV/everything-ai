# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson start-using-ai-tools. Route: course/route/lessen/stam-1-basis.yaml. Checks and the made-up next words: checks.py.
# termen:allow genai

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("start-using-ai-tools", "en")
    C = L.checks   # the made-up next words for the picture


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about the four circles.
    """)
    return


@app.cell(hide_code=True)
def quick_check_sketch():
    quick_sketch = L.choice(
        "quick-sketch",
        question="An app turns your rough sketch into a finished painting, with a neural network that learned from many paintings. What is the smallest circle it fits in?",
        options={"ml": "Machine learning", "dl": "Deep learning", "genai": "Generative AI"},
        correct="genai",
        why={
            "ml": "It fits in machine learning, but there are smaller circles: it has a neural network, and it makes something. Go through the three questions of [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en) again.",
            "dl": "It fits in deep learning, but there is a smaller circle. Does it make something new, or only sort and predict? See [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en).",
            "genai": "It learned from examples with a neural network, and it makes something new: a painting. So it is generative AI.",
        },
    )
    quick_sketch
    return (quick_sketch,)


@app.cell(hide_code=True)
def quick_check_question():
    quick_question = L.choice(
        "quick-question",
        question="Which question tells you whether a deep learning system is also generative AI?",
        options={
            "examples": "Did it learn from examples?",
            "layers": "Is it a neural network with many layers?",
            "new": "Does it make something new, such as text or pictures?",
        },
        correct="new",
        why={
            "new": "Generative AI is the part of deep learning that makes something new.",
            "examples": "Every deep learning system learned from examples, so this question cannot tell the two apart. Look at the last question of the recipe in [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en).",
            "layers": "Every deep learning system is a neural network with many layers, so this question cannot tell the two apart. Look at the last question of the recipe in [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en).",
        },
    )
    quick_question
    return (quick_question,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Chat programs such as ChatGPT, Claude and Gemini (well-known examples in October 2026) write emails, explain homework, plan trips and write code. Hundreds of millions of people use them in 2026, and they can help you in this course too. But they also get things wrong, and a wrong answer often comes with no warning. If you know how they work, you know when you can rely on an answer and when you should check it.

    **An everyday example.** Picture someone who has read an enormous amount and answers every question straight away, fluently and without hesitating, even when they are not sure. You would happily let them help you write a letter. But before you take a medicine on their word, you read the label. That is only a comparison: a chat program does not read or doubt the way a person does. What it does instead, you see below.
    """)
    return


@app.cell(hide_code=True)
def predict_sound():
    guess_sound = L.guess(
        "how-it-sounds",
        question="A chat program gives you an answer that is wrong. How do you think that answer sounds?",
        options={
            "unsure": "Less sure than usual: it hesitates.",
            "same": "Just as sure as a right answer.",
            "warns": "It warns you that it may be wrong.",
        },
        reveal="**Often just as sure as a right answer.** How sure an answer sounds tells you little about whether it is right. In this lesson you see why, and what you can do about it.",
    )
    guess_sound
    return (guess_sound,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **AI assistant** | a chat program that answers questions, explains, drafts texts and writes code; it learned to do that from a huge amount of text |

    After this, we just say *assistant*.

    ### Word by word

    Behind the chat window is a model: the part that learned, from a huge amount of text, how text goes on. It looks at the text so far: your question and the answer up to now. Then it picks a likely next word, adds it, and does the same again, word by word, until the answer is done. A model like that is called a language model. At the end of the basics you build a small one yourself, and there you see the idea from the inside.

    It does not look the answer up in a list of facts: what it learned is how text usually goes on. That is also a simplification: the model really works with pieces of words, and a later lesson shows how. The main point stands: **likely is not the same as true.** For a question that many texts answer, the likely answer is usually right. For something few texts mention, such as a fact about your own town or an exact number, or something that happened after the model learned its text, it can still write a plausible answer, just as fluently.
    """)
    return


@app.cell(hide_code=True)
def picture_next_words():
    mo.vstack([
        draw.next_word_bars(
            "The windmill in our town was built in", C.NEXT_WORDS["en"], picked=0, picked_label="written in the answer",
            title="How likely each next word is"),
        mo.md(r"""
        Each **blue bar** shows how likely the model thinks a next word is; the rest is spread over many other words. The numbers are made up for this example, but this is how it works: several years sound likely, the model writes one of them, and nothing in the answer tells you that it guessed.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_good_at():
    mo.vstack([
        mo.md(r"""
        ### What an assistant is good at, and what to watch out for

        **Good at:** explaining something in other words, giving examples, writing a first draft of a text, coming up with ideas, rewriting in another tone, writing simple code.

        **Watch out for:** exact facts about rare things; numbers, dates, quotes and sources; recent events; and everything where a mistake costs a lot, such as health, money, legal matters, and code that deletes or sends something.

        The line between the two is not sharp. An assistant can explain a hard idea well and get a simple date wrong in the same answer.
        """),
        L.watch_out("An assistant can do more than the model inside it: some assistants search the web or read a file you give them. That helps, but the language model still writes the answer word by word, and it may say that a page says something that is not on it."),
        mo.md(r"""
        ### Recipe: check what matters

        1. **What will you do with the answer?** Only use it as an idea or a draft that you go through yourself? Then you can use it as it is. Will you pass it on, rely on it or act on it? Go on to step 2.
        2. **Which parts does that depend on?** Pick out the facts: names, numbers, dates, quotes, sources, and what a piece of code does.
        3. **Check those parts somewhere else,** in a source you trust: the original document, an official website, a book, or by trying the code on a copy, where a mistake does no harm. Not with the same assistant: it can confirm its own mistake.
        4. **Could not check it?** Leave it out, or say that it is not checked.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    You write a short piece about the windmill in your town for the school paper, and you ask an assistant for a start. This is a made-up answer, but a typical one:

    > Here is a start for your piece. The windmill on the Dijkweg was built in 1782 and is one of the oldest in the region. According to the local history society, it was moved twice. On a summer evening it is a beautiful sight. It is open to visitors every Saturday. A title could be: *Still turning after all these years*.

    1. **What will you do with it?** You publish it, and readers will take it as true. So go on. (Had you only asked for ideas for a title, you would stop here: ideas you choose from yourself need no checking.)
    2. **Which parts does that depend on?** The facts: built in 1782, one of the oldest in the region, moved twice according to the history society, open every Saturday. Not facts: "a beautiful sight" is an opinion you keep or drop, and the title is your own choice.
    3. **Check them somewhere else.** The year and the moves on the history society's own website, or in the town archive. The opening days on the mill's own website, or on the sign at the door. "One of the oldest" by comparing with a list of the mills in the region.
    4. **Could not check it?** Say the history society's website says nothing about moving the mill. Then that sentence goes, until you find a source.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_check():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Which parts are facts?

        You plan a day at a museum with your grandmother, who uses a wheelchair, and you ask an assistant for help. Its answer has the six parts below. Step 1 of the recipe is done: you will rely on this plan on the day itself. Now do step 2.
        """),
        L.task("Put each part of the answer in the right group."),
    ])
    return


@app.cell(hide_code=True)
def exercise_check():
    sort_check = L.sort("what-to-check")
    sort_check
    return (sort_check,)


@app.cell(hide_code=True)
def check_what_to_check(sort_check):
    result_check = L.check("what-to-check", sort_check)
    result_check
    return (result_check,)


@app.cell(hide_code=True)
def hints_check():
    L.hints(
        "what-to-check",
        [
            "Step 2 of the recipe: pick out the facts. Names, numbers, times, prices and what is possible on the day.",
            "Two parts are not facts at all: they are suggestions, like the title in the worked example.",
            "Ask of each part: would the day go wrong if this were not true? Four parts are like that.",
        ],
        why="A fact the day depends on: **Opening hours**, **Wheelchair**, **Tickets** and **Train**. Not a fact: **Tip** and **Idea**, suggestions you decide on yourself.",
    )
    return


@app.cell(hide_code=True)
def text_how_check():
    mo.md(r"""
    ### Exercise 2. How do you check?
    """)
    return


@app.cell(hide_code=True)
def choice_how_check_cell():
    choice_how_check = L.choice(
        "how-check",
        question="You want to be sure that every room of the museum can be reached by lift. What is the best way to check?",
        options={
            "again": "Ask the same assistant: \"Are you sure about the lift?\"",
            "source": "Look it up on the museum's own website, or call the museum.",
            "link": "Ask the assistant for a link, and trust the answer because there is a link.",
        },
        correct="source",
        why={
            "again": "The same assistant is not a second source: it can repeat its mistake, or change its answer, and sound just as sure either way. Step 3 of the recipe: who knows this for certain?",
            "source": "The museum knows its own building. That is checking at the source, step 3 of the recipe.",
            "link": "A link alone proves nothing: the page may say something else, or not exist at all. Step 3 of the recipe: who knows this for certain?",
        },
    )
    choice_how_check
    return (choice_how_check,)


@app.cell(hide_code=True)
def text_explain():
    mo.md(r"""
    ### Exercise 3. Why so sure?

    Write your own answer first. The model answer you can compare it with is an example answer from the course, not an answer from an AI model.
    """)
    return


@app.cell(hide_code=True)
def explain_why_sure():
    explain_sure = L.explain(
        "why-sure",
        question="Why can an assistant write a wrong year just as fluently as a right one? Write one or two sentences.",
        model_answer="The language model inside the assistant writes the answer word by word, each time picking a word that is likely after the text so far. A year that sounds likely is not always the right year, and picking a likely word is not checking a fact, so a wrong year comes out just as fluently.",
        points=[
            "it picks a likely next word, word by word",
            "likely is not the same as true",
            "picking a likely word is not checking a fact",
        ],
    )
    explain_sure
    return (explain_sure,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en) you put a chat program like this in the smallest circle: generative AI. Now you know what is inside an AI assistant: a language model that writes word by word. At the end of the basics you build a small language model yourself, and you see where "likely but wrong" comes from; a later lesson gives that its own name. The recipe from this lesson stays useful throughout the course, also when an assistant helps you with your own code.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_how_check, explain_sure, result_check):
    L.wrap_up(result_check, choice_how_check, explain_sure)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
