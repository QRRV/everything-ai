# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson start-ai-ml-genai. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("start-ai-ml-genai", "en")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about the lesson before this one. They stop nothing: if one goes wrong, the explanation tells you where to look again.
    """)
    return


@app.cell(hide_code=True)
def quick_check_run():
    quick_run = L.choice(
        "quick-run",
        question="How do you run a code cell?",
        options={
            "keys": "Press **Shift + Enter**, or click the ▶ button next to the cell.",
            "reopen": "Close the lesson and open it again.",
            "word": "Type `run` at the bottom of the cell.",
        },
        correct="keys",
        why={
            "keys": "**Shift + Enter** runs the cell you are in, and its result appears right under the code.",
            "reopen": "Opening a lesson runs the cells that come with it, but not the ones you changed. To run one cell, press **Shift + Enter**. See [How a lesson works](/course/open/start-how-this-course-works?lang=en).",
            "word": "`run` is not a Python word, so Python would stop with an error. You run a cell with **Shift + Enter**. See [How a lesson works](/course/open/start-how-this-course-works?lang=en).",
        },
    )
    quick_run
    return (quick_run,)


@app.cell(hide_code=True)
def quick_check_not_yet():
    quick_not_yet = L.choice(
        "quick-not-yet",
        question="A check says *Not yet*. What is the best thing to do?",
        options={
            "read": "Read what the check says, change your answer, and look again.",
            "solution": "Open the solution straight away.",
            "restart": "Start the lesson over.",
        },
        correct="read",
        why={
            "read": "A check says what is not right yet and what you can do about it. Trying again with that is how you learn the most.",
            "solution": "The solution is there for when you are really stuck. First try what the check suggests, then the hints one at a time.",
            "restart": "One wrong answer is no reason to start over. Read what the check says and try again.",
        },
    )
    quick_not_yet
    return (quick_not_yet,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    "AI writes your emails." "Machine learning catches fraud." "A deep learning model found the tumor." You hear these words every day, and they are often used as if they mean the same thing. They do not. When someone says that an app "uses AI", you cannot tell yet what it does or how it was made.

    **An everyday example.** Vehicle, car, electric car. Every electric car is a car, and every car is a vehicle, but a bicycle is a vehicle and not a car. When someone just says "vehicle", you do not know which one they mean. The four words in this lesson work the same way.
    """)
    return


@app.cell(hide_code=True)
def predict_learning():
    guess_learning = L.guess(
        "every-ai-learns",
        question="Do you think every AI system learned what it does from examples?",
        options={
            "yes": "Yes, that is what makes it AI.",
            "no": "No, some AI follows rules that people wrote.",
            "new": "Only the AI from the last few years.",
        },
        reveal="**No.** Much AI follows rules that people wrote, such as a chess program that scores positions with rules from chess players. Learning from examples is *one* way to build AI, called machine learning. Today it is by far the most common way, which is why the words get mixed up so easily.",
    )
    guess_learning
    return (guess_learning,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    The four words are circles inside each other. Each circle is a part of the circle around it.

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **artificial intelligence** (AI) | the field that builds computer systems for tasks we used to think needed human intelligence, such as recognizing images, understanding language or planning |
    | **machine learning** (ML) | the part of AI where the computer finds a rule in examples, instead of a person writing the rule |
    | **deep learning** | machine learning with *neural networks* that have many *layers*; you will build one yourself later in the course |
    | **generative AI** | AI that makes something new, such as text, images, sound or code, instead of only sorting or predicting |

    After this lesson we just say AI and ML.
    """)
    return


@app.cell(hide_code=True)
def picture_circles():
    mo.Html("""
    <figure class="ea-figure" style="margin:0.5rem 0 1rem;max-width:40rem">
    <svg viewBox="0 0 640 370" role="img" aria-labelledby="circles-title" style="width:100%;height:auto;font-family:var(--ea-font-ui)">
      <title id="circles-title">Four circles inside each other: AI, machine learning, deep learning and generative AI</title>
      <g style="stroke:var(--ea-ink-2);stroke-width:1.5">
        <ellipse cx="320" cy="190" rx="312" ry="170" style="fill:var(--ea-ink);fill-opacity:0.03"/>
        <ellipse cx="320" cy="228" rx="248" ry="130" style="fill:var(--ea-ink);fill-opacity:0.05"/>
        <ellipse cx="320" cy="266" rx="178" ry="92" style="fill:var(--ea-ink);fill-opacity:0.07"/>
        <ellipse cx="320" cy="303" rx="88" ry="55" style="fill:var(--ea-ink);fill-opacity:0.10"/>
      </g>
      <g style="fill:var(--ea-ink);text-anchor:middle">
        <text x="320" y="50" style="font-weight:680;font-size:19px">AI</text>
        <text x="320" y="72" style="font-size:14px;fill:var(--ea-ink-2)">for example a route planner</text>
        <text x="320" y="128" style="font-weight:680;font-size:19px">Machine learning</text>
        <text x="320" y="150" style="font-size:14px;fill:var(--ea-ink-2)">for example a house price estimate</text>
        <text x="320" y="203" style="font-weight:680;font-size:19px">Deep learning</text>
        <text x="320" y="225" style="font-size:14px;fill:var(--ea-ink-2)">for example face unlock</text>
        <text x="320" y="292" style="font-weight:680;font-size:17px">Generative AI</text>
        <text x="320" y="315" style="font-size:13px;fill:var(--ea-ink-2)">for example an</text>
        <text x="320" y="332" style="font-size:13px;fill:var(--ea-ink-2)">image generator</text>
      </g>
    </svg>
    <figcaption class="ea-needs">Every system sits in the smallest circle it fits in, and with that also in every circle around it.</figcaption>
    </figure>
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        This is the common way to draw the circles. The borders are a little fuzzier in real life, but this picture is enough for the whole course.
        """),
        L.watch_out("In the news, \"AI\" often means generative AI only: a chat assistant or an image generator. In this course, AI means the whole big circle."),
        mo.md(r"""
        ### Recipe

        To find the smallest circle a system fits in, ask three questions, in this order:

        1. **Did people write its rules, or did the computer find them in examples?** People wrote them: stop, it is in the AI circle.
        2. **Is it a neural network with many layers?** No: stop, it is machine learning.
        3. **Does it make something new, such as text, images or sound?** No: it is deep learning. Yes: it is generative AI.

        Some real systems mix rules and learning. Then look at the part that does the main work.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    **Photo search.** Your phone finds every photo of your dog. It learned what a dog looks like from millions of photos marked "dog" or "no dog", with a neural network of many layers.

    1. **Rules or examples?** It learned from photos marked "dog" or "no dog": examples. So it is inside machine learning.
    2. **A neural network with many layers?** Yes, the description says so. So it is inside deep learning.
    3. **Something new?** No, it finds photos that already exist. So it is not generative AI.

    The smallest circle is **deep learning**. Photo search also sits in the two circles around it: it is machine learning, and it is AI.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_circles():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Five systems, four circles

        Each system below gets one circle. Use the recipe: the three questions, in order.
        """),
        L.task("Put each system in the **smallest** circle it fits in."),
    ])
    return


@app.cell(hide_code=True)
def exercise_circles():
    sort_circles = L.sort("circles")
    sort_circles
    return (sort_circles,)


@app.cell(hide_code=True)
def check_circles(sort_circles):
    result_circles = L.check("circles", sort_circles)
    result_circles
    return (result_circles,)


@app.cell(hide_code=True)
def hints_circles():
    L.hints(
        "circles",
        [
            "Start with question 1 for every system: which ones follow rules that people wrote? Those stay in the big AI circle.",
            "Question 2: which of the others say *neural network*? The ones without stop at machine learning.",
            "Question 3: the chat assistant and the music maker make something new. The plant app only names a plant.",
        ],
        why="Plant app: **deep learning**. Spam filter: **machine learning**. Chat assistant: **generative AI**. Chess program: **AI**. Music maker: **generative AI**.",
    )
    return


@app.cell(hide_code=True)
def text_inside():
    mo.md(r"""
    ### Exercise 2. Inside each other
    """)
    return


@app.cell(hide_code=True)
def choice_inside_cell():
    choice_inside = L.choice(
        "inside",
        question="Which sentence is right?",
        options={
            "all-learn": "Every AI system learns from examples.",
            "genai-ml": "Every generative AI system is also machine learning.",
            "same": "Deep learning and generative AI mean the same thing.",
        },
        correct="genai-ml",
        why={
            "all-learn": "Not every AI system learns: the chess program follows rules that people wrote, and it is AI too. Machine learning is the part of AI that learns from examples.",
            "genai-ml": "Generative AI is the smallest circle. It lies inside deep learning, which lies inside machine learning, so it learned from examples too.",
            "same": "Generative AI is the part of deep learning that makes something new. The plant app is deep learning, but it makes nothing new.",
        },
    )
    choice_inside
    return (choice_inside,)


@app.cell(hide_code=True)
def text_ask():
    mo.md(r"""
    ### Exercise 3. Ask the right questions
    """)
    return


@app.cell(hide_code=True)
def explain_ask():
    explain_questions = L.explain(
        "ask",
        question="A news article says: \"This app uses AI.\" Which questions would you ask to find out which circle it belongs in? Write them down.",
        model_answer="**Did people write its rules, or did it learn them from examples?** If it learned: **is it a neural network with many layers?** And: **does it make something new, such as text or pictures, or does it sort and predict?**",
        points=[
            "you ask whether it learned from examples or follows rules that people wrote",
            "you ask whether it is a neural network with many layers",
            "you ask whether it makes something new",
        ],
    )
    explain_questions
    return (explain_questions,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [How a lesson works](/course/open/start-how-this-course-works?lang=en) you learned to find your way around a lesson; from now on, every lesson is about AI itself. The next lesson zooms in on the first question of the recipe: writing rules versus learning from examples. Later in the course you build each circle yourself: a model that learns from examples, then your own neural network, and at the end of the main route a small generative model that writes text.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_inside, explain_questions, result_circles):
    L.wrap_up(result_circles, choice_inside, explain_questions)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
