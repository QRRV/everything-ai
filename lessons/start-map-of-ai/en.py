# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson start-map-of-ai. Route: course/route/lessen/stam-1-basis.yaml. Checks and the numbers about the route: checks.py.
# termen:allow genai

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("start-map-of-ai", "en")
    C = L.checks   # the numbers about the route, counted from the route data
    GPT = C.goal("doel-eigen-gpt")


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
def quick_check_fraud():
    quick_fraud = L.choice(
        "quick-fraud",
        question="A bank's fraud detector learned from millions of payments, each marked as fraud or not, with a neural network of many layers. It only says \"suspicious\" or \"fine\". What is the smallest circle it fits in?",
        options={"ml": "Machine learning", "dl": "Deep learning", "genai": "Generative AI"},
        correct="dl",
        why={
            "ml": "It fits in machine learning, but there is a smaller circle: it is a neural network with many layers, so it is deep learning. See [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en).",
            "dl": "It learned from examples with a neural network of many layers, and it makes nothing new. So it is deep learning.",
            "genai": "It only says \"suspicious\" or \"fine\" about payments that already exist; it makes nothing new. So it is deep learning, not generative AI. See [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en).",
        },
    )
    quick_fraud
    return (quick_fraud,)


@app.cell(hide_code=True)
def quick_check_order():
    quick_order = L.choice(
        "quick-order",
        question="Which order is right, from the biggest circle to the smallest?",
        options={
            "swapped": "AI, deep learning, machine learning, generative AI",
            "reversed": "Generative AI, deep learning, machine learning, AI",
            "right": "AI, machine learning, deep learning, generative AI",
        },
        correct="right",
        why={
            "right": "AI is the biggest circle, and generative AI the smallest, inside deep learning.",
            "swapped": "Deep learning is a part of machine learning: machine learning with neural networks of many layers. So machine learning is the bigger circle. See [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en).",
            "reversed": "That is the order from the smallest circle to the biggest. The biggest circle is AI.",
        },
    )
    quick_order
    return (quick_order,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In an earlier lesson you met four words for parts of AI. But AI is much more than those four circles. It also covers programs that plan a trip or play chess, robots that learn to pick things up, systems that turn speech into text, and laws about what AI may do. Nobody learns all of that at once, and not in just any order: a lot of it builds on other parts. So before you set off, you need a map: what is there, what comes before what, and which way you can go.

    **An everyday example.** Learning to cook. Everyone starts with the same basics: cutting, heating, tasting. Then you pick a direction, such as baking bread or Italian food, and in each direction you go from easy dishes to hard ones. A good cookbook does not open with a soufflé, and it lets you skip the bread chapter if all you want is pasta.
    """)
    return


@app.cell(hide_code=True)
def predict_size():
    guess_size = L.guess(
        "how-big",
        question="How many topics do you think the field of AI has, from the basics to the newest research?",
        options={
            "50": "About 50",
            "200": "About 200",
            "600": "More than 600",
        },
        reveal="**More than 600.** When this course was planned, the whole field was mapped out: 623 topics in 27 fields, from Python and math to robots and law, and each topic has smaller parts of its own. Nobody needs all of them for one goal. That is why the course is laid out as a map, with routes through it.",
    )
    guess_size
    return (guess_size,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Picture the course as a tree. That is only a comparison, but it helps: there is one trunk, the **foundation**, and every other part of AI grows out of it as a branch. Every lesson also has a level, and a route says in which order you take the lessons.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **branch** | one area of the course, such as language or computer vision (AI for pictures and video), with its own lessons and its own project; every branch grows from the **foundation**, the branch with math, Python, data, machine learning and deep learning that everyone takes |
        | **level** | the label on a lesson that says how advanced it is: **basics**, **intermediate** or **expert** |
        | **route** | an order of lessons through the course; the **main route** goes from your first line of Python to a model of your own that writes text, and a research paper that you reproduce yourself; a **goal route** takes only the lessons that one goal builds on |

        Three examples, one of each: robotics is a branch; *basics* is a level; and *I want my own GPT as fast as possible*, a goal on the start page, is a route.
        """),
        L.watch_out("In a game, a level is a stage that opens only when you finish the one before. Here a level is just a label that says how advanced a lesson is: no lesson is locked."),
    ])
    return


@app.cell(hide_code=True)
def text_branches():
    mo.md(rf"""
    The course groups the 27 fields of AI into {C.N_BRANCHES} branches. The main route goes through three of them, at all three levels: the foundation, language and research. The figure counts the lessons of the main route per level and per branch.
    """)
    return


@app.cell(hide_code=True)
def picture_route():
    mo.vstack([
        draw.route_grid(
            ["Basics", "Intermediate", "Expert"],
            ["Foundation", "Language", "Research", "Other branches"],
            C.grid(), unit=("lesson", "lessons"), arrows=3,
            title="The main route: lessons per level and per branch"),
        mo.md(r"""
        Read each row from left to right, then go down a row. A darker box holds more lessons. The dashed column counts the few lessons the main route borrows from other branches; they come in between, where you need them. So at each level you go mostly from the foundation to language and then to research.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(rf"""
    So you do not have to pick a branch now: the main route takes you through the foundation, language and research. The other {C.N_BRANCHES - 3} branches, such as computer vision and robotics, come later; the start page shows them all under *The whole map*. Each of them builds on the main route, and most of their first lessons need only its basics level.

    ### Recipe

    To pick your route, go through these questions in order:

    1. **No goal of your own yet?** Follow the main route. On the start page, *Continue* takes you to your next lesson.
    2. **A goal of your own?** Pick it on the start page under *Or follow a goal*. You then see only the lessons that goal builds on, in the order of the main route.
    3. **Want to go deeper?** Switch on *Show extra lessons*. You never need an extra lesson for the lessons after it.
    4. **Opened a lesson further ahead?** Look at the top, under *Before you start*: it lists the lessons it builds on, with a tick for each one you have done.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(rf"""
    ### Worked example

    **Case 1: no goal yet.** You want to learn AI from the start, and you have no goal of your own yet. Question 1 says it all: follow the main route, and *Continue* on the start page takes you to your next lesson. Questions 3 and 4 still help along the way.

    **Case 2: a goal.** You want to build a GPT as fast as possible: a model that writes text. You build it yourself at the intermediate level.

    1. **No goal of your own yet?** You do have one, so on to question 2.
    2. **A goal of your own?** Yes. On the start page you pick *I want my own GPT as fast as possible*. The start page counts the lessons this goal builds on: **{GPT["lessons"]} lessons**, instead of the {GPT["place"]} lessons up to the GPT in the main route (with *Show extra lessons* off). Look at which ones they are:
        - **Levels:** {GPT["levels"]["basis"]} at basics and {GPT["levels"]["gevorderd"]} at intermediate. None at expert: a first GPT does not need those.
        - **Branches:** {GPT["foundation"]} from the foundation, and the rest mostly from the language branch.
    3. **Want to go deeper?** Not now: you want your GPT fast. Leave *Show extra lessons* off.
    4. **Opened a lesson further ahead?** Not needed: the goal route gives you every lesson in order.

    So the goal route skips {GPT["place"] - GPT["lessons"]} lessons that a first GPT does not build on.
    """)
    return


@app.cell(hide_code=True)
def text_pick_route():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Pick a route

    Use the recipe, as in case 2 of the worked example.
    """)
    return


@app.cell(hide_code=True)
def choice_pick_route_cell():
    choice_pick_route = L.choice(
        "pick-route",
        question="Sam wants to understand attention, an idea you meet later in the course, as fast as possible. What should Sam do?",
        options={
            "jump": "Open the attention lesson straight away.",
            "goal": "Pick *I want to understand attention* on the start page, and follow that route.",
            "all": "Take the whole main route first, up to attention.",
        },
        correct="goal",
        why={
            "goal": "That route has exactly the lessons attention builds on, in the right order, and nothing else.",
            "jump": "That lesson builds on many lessons before it: under *Before you start* you would see them, without ticks. Look at question 2 of the recipe again.",
            "all": "That works, but it takes longer than needed: before attention, the main route also has lessons that attention does not build on. Look at question 2 of the recipe again.",
        },
    )
    choice_pick_route
    return (choice_pick_route,)


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Branch, level or route?

        Each item below is a branch, a level or a route, like the three examples under the table with the new words.
        """),
        L.task("Put each item in the right group."),
    ])
    return


@app.cell(hide_code=True)
def exercise_kinds():
    sort_kinds = L.sort("kinds")
    sort_kinds
    return (sort_kinds,)


@app.cell(hide_code=True)
def check_kinds(sort_kinds):
    result_kinds = L.check("kinds", sort_kinds)
    result_kinds
    return (result_kinds,)


@app.cell(hide_code=True)
def hints_kinds():
    L.hints(
        "kinds",
        [
            "A branch is a part of AI, a level says how advanced a lesson is, and a route is an order of lessons.",
            "Two items are labels on lessons: those are the levels. Which two items are lists of lessons to take one after the other?",
            "Computer vision and speech are parts of AI, each with lessons of its own.",
        ],
        why="Branch: **Computer vision** and **Speech, audio and music**. Level: **Intermediate** and **Expert**. Route: **I want to understand attention** and **The main route**.",
    )
    return


@app.cell(hide_code=True)
def text_foundation():
    mo.md(r"""
    ### Exercise 3. How the branches hang together
    """)
    return


@app.cell(hide_code=True)
def choice_foundation_cell():
    choice_foundation = L.choice(
        "foundation",
        question="Which sentence about the branches is right?",
        options={
            "separate": "Each branch is a course of its own: you can start in any branch without anything before it.",
            "levels": "There are three branches: basics, intermediate and expert.",
            "grow": "Every branch grows from the foundation: its lessons build on lessons from the foundation.",
        },
        correct="grow",
        why={
            "grow": "Every part of AI uses the same basics, so every branch starts where the foundation has brought you.",
            "separate": "A first lesson on computer vision, for example, stores a picture as numbers with Python; without the foundation you could not follow it. Look at the table with the new words again.",
            "levels": "Basics, intermediate and expert are labels that say how advanced a lesson is. Look at the table with the new words again: what is a branch?",
        },
    )
    choice_foundation
    return (choice_foundation,)


@app.cell(hide_code=True)
def text_further():
    mo.md(r"""
    ### Going further

    Not needed for this lesson, but worth a look: on the start page, pick *I want to train my first CNN* under *Or follow a goal*. A CNN is a neural network for pictures, which you build at the intermediate level. At which levels are the lessons of that route? Then pick *The whole route* again to get the main route back.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en) you met four circles; they are a small part of the map. From now on, every lesson sits somewhere on this map: the top of a lesson shows its place in the route and its level, and the start page shows the whole route. The next lesson also belongs to the start: how you use AI tools such as a chat assistant, with your eyes open.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_foundation, choice_pick_route, result_kinds):
    L.wrap_up(choice_pick_route, result_kinds, choice_foundation)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
