# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson start-your-own-assistant. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("start-your-own-assistant", "en")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about the lessons before this one.
    """)
    return


@app.cell(hide_code=True)
def quick_check_monday():
    quick_monday = L.choice(
        "quick-monday",
        question="An assistant tells you that the museum is also open on Mondays. You want to go on a Monday. What do you do?",
        options={
            "sure": "Ask the assistant whether it is sure.",
            "source": "Look it up on the museum's own website.",
            "trust": "Go: it said so without any doubt.",
        },
        correct="source",
        why={
            "sure": "The same assistant is not a second source: it can repeat its mistake and sound just as sure. See the recipe in [Using AI tools with your eyes open](/course/open/start-using-ai-tools?lang=en).",
            "source": "The museum knows its own opening days. That is checking at the source.",
            "trust": "How sure an answer sounds tells you little about whether it is right. See the recipe in [Using AI tools with your eyes open](/course/open/start-using-ai-tools?lang=en).",
        },
    )
    quick_monday
    return (quick_monday,)


@app.cell(hide_code=True)
def quick_check_outcome():
    quick_outcome = L.choice(
        "quick-outcome",
        question="What does a check in this course look at?",
        options={
            "form": "Whether you wrote it exactly like the course's solution.",
            "time": "How long you took.",
            "outcome": "The outcome of your answer.",
        },
        correct="outcome",
        why={
            "form": "A check does not compare your text with the course's solution: `5 + 5` counts just as well as `3 + 7`. See [How a lesson works](/course/open/start-how-this-course-works?lang=en).",
            "time": "A check does not time you. See [How a lesson works](/course/open/start-how-this-course-works?lang=en).",
            "outcome": "A check looks at the outcome, so a different answer with the right outcome counts as right too.",
        },
    )
    quick_outcome
    return (quick_outcome,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You may already use an AI assistant. While you learn, it can be a great help: it never gets tired of your questions, and it explains the same thing in ten different ways. But it can also do the thinking for you. If it hands you every answer, you finish the lesson and have learned nothing. And it can be wrong, as you saw in [Using AI tools with your eyes open](/course/open/start-using-ai-tools?lang=en).

    **An everyday example.** Picture a driving instructor with their own pedals. You only learn to drive if they let you steer. If they take over at every bend, you arrive safely but still cannot drive. A good instructor does not grab the wheel, but says: "Look in your mirror."
    """)
    return


@app.cell(hide_code=True)
def predict_help():
    guess_help = L.guess(
        "which-help",
        question="You are stuck on an exercise. Which help do you think makes you learn the most?",
        options={
            "answer": "The full answer, so you can study it.",
            "hint": "A hint, so you find the rest yourself.",
            "nothing": "No help at all: keep trying alone.",
        },
        reveal="**Usually a hint.** Studying a full answer helps too, which is why every lesson shows a worked example first. But once you are stuck, a hint leaves the last step to you, and a step you take yourself tends to stick better. The course's own hints work the same way: they open one at a time, with the solution last.",
    )
    guess_help
    return (guess_help,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    You can let your own assistant help you with this course, as long as it helps the way a good instructor does.

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **AGENTS.md** | a file in the course folder with instructions for an AI assistant: how the course is built, and how to help you without giving the answers away |

    ### How your assistant knows the course

    Some assistants work inside a folder on your computer, such as Claude Code, Codex and Cursor (examples in October 2026). Open one of them in the course folder: the folder where you started the course. In October 2026 most of them read AGENTS.md by themselves; if yours does not, ask it to read AGENTS.md first. It can then also see where you are: your own copy of each lesson and your progress, including what you typed in the exercise boxes, are in the folder *work*. A chat program in your browser does not see the folder. With a chat program, tell it which lesson and exercise you are on, and paste in the exercise.

    AGENTS.md asks your assistant, among other things, not to give the answer away, to use only what you have learned so far, and to answer in your language. If you ask for the answer outright, it may still give it: that choice stays yours. And an assistant does not always follow instructions, which is one more reason to compare what it says with the lesson.
    """)
    return


@app.cell(hide_code=True)
def picture_flow():
    mo.vstack([
        draw.flow(
            [("You", ["you ask, take the last", "step, and the check", "says if it is right"]),
             ("Your assistant", ["explains and hints,", "can be wrong"]),
             ("The course folder", ["AGENTS.md", "the lessons", "work: your progress"])],
            forward=["your question", "reads"],
            back=["a hint,\nan example or\na question back"],
            title="Who does what when you learn with an assistant"),
        mo.md(r"""
        With an assistant in the course folder: your question goes to the assistant, it reads AGENTS.md and your work in the folder, and gives you something back that keeps the thinking with you. Whether your answer is right is still decided by the check in the lesson.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: learn with an assistant

    1. **Let it see the course.** Open it in the course folder, or tell it which lesson and exercise you are on and paste in the exercise.
    2. **Ask for help, not for the answer.** For example: "Explain this with another example", "Give me one hint", or "Ask me a question that points the way."
    3. **Compare what it says with the lesson.** Does it match? Does it use words the course has not taught you yet? Then ask it to explain without them, or to say in which lesson the course teaches them.
    4. **Take the step yourself, and let the check decide.** Your answer is right when the check under the exercise says *Correct*, not when the assistant says so.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    You are at exercise 1 of [How a lesson works](/course/open/start-how-this-course-works?lang=en): the sum in the box should give 10. You typed `3 + 5`, and the check says *Not yet*. Your assistant is open in the course folder.

    1. **Let it see the course.** It is open in the course folder, so it can see in your progress what you typed in the box, and look up what the check says about it. (With a chat program in your browser, you would type the lesson and the exercise yourself, and paste in `3 + 5` and what the check says.)
    2. **Ask for help, not for the answer.** You type: *"I am at exercise 1 of How a lesson works, and the check says Not yet. Give me one hint, not the answer."* You do not type *"What should I put in the box?"*: then the assistant takes the last step for you. It answers: *"Look at what your sum gives now. How much is missing to reach 10?"*
    3. **Compare what it says with the lesson.** The hint matches the lesson: the check's message also says what your sum gives. It uses no new words. Suppose it had said *"Store the sum in a variable first."* The course has not taught you what a variable is yet, so you would ask it to explain without that word, or to say in which lesson the course teaches it.
    4. **Take the step yourself, and let the check decide.** You see that `3 + 5` gives 8, so you are 2 short. You change the 5 into a 7, and the check under the box says *Correct*.

    A second case: suppose the assistant had said *"Your answer is right"* while the check still said *Not yet*. Then you go by the check: it is made for exactly this exercise. Back to step 2, and ask the assistant what the check's message means.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ask():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Which requests keep the thinking with you?

        Six requests you could send to an assistant. Look at step 2 of the recipe, and at the request in the worked example that you do not type.
        """),
        L.task("Put each request in the right group."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ask():
    sort_ask = L.sort("ask-kinds")
    sort_ask
    return (sort_ask,)


@app.cell(hide_code=True)
def check_ask(sort_ask):
    result_ask = L.check("ask-kinds", sort_ask)
    result_ask
    return (result_ask,)


@app.cell(hide_code=True)
def hints_ask():
    L.hints(
        "ask-kinds",
        [
            "Look at step 2 of the recipe: which kinds of help does it name?",
            "For each request, ask: after the assistant answers, is there still a step left for you?",
            "Three requests leave a step for you, and three do it for you. *Fill it in* is one that does it for you.",
        ],
        why="Keeps the thinking with you: **Another example**, **One hint** and **A question back**; after these, the last step is still yours. Takes the thinking away: **Fill it in**, **Just the answer** and **Make it right**; these hand you a finished answer.",
    )
    return


@app.cell(hide_code=True)
def text_trust_check():
    mo.md(r"""
    ### Exercise 2. The assistant and the check disagree
    """)
    return


@app.cell(hide_code=True)
def choice_trust_check_cell():
    choice_trust_check = L.choice(
        "trust-check",
        question="Your assistant says your answer is right, but the check under the exercise says *Not yet*. What do you do?",
        options={
            "assistant": "Go by the assistant: it explains things so well.",
            "check": "Go by the check, and ask the assistant what its message means.",
            "skip": "Skip the exercise: the check must be wrong.",
        },
        correct="check",
        why={
            "assistant": "Explaining well and being right are two different things: an assistant can be wrong and sound just as sure. Look at step 4 of the recipe again.",
            "check": "The check is made for exactly this exercise and looks at the outcome of your answer. An assistant can be wrong and still sound sure.",
            "skip": "A check can contain a mistake too; if you are sure it does, report it with the link at the bottom of the lesson. Skipping teaches you nothing. Look at step 4 of the recipe again.",
        },
    )
    choice_trust_check
    return (choice_trust_check,)


@app.cell(hide_code=True)
def text_new_word():
    mo.md(r"""
    ### Exercise 3. A word you do not know yet
    """)
    return


@app.cell(hide_code=True)
def choice_new_word_cell():
    choice_new_word = L.choice(
        "new-word",
        question="Your assistant explains a step with a word the course has not taught you yet. What do you do?",
        options={
            "nod": "Nod along: it will become clear by itself later.",
            "stop": "Stop using an assistant: it is not meant for beginners.",
            "ask": "Ask it to explain without that word, or to say in which lesson the course teaches it.",
        },
        correct="ask",
        why={
            "nod": "If you do not understand one word, the rest of the explanation hangs in the air too. Look at step 3 of the recipe again.",
            "stop": "An assistant can help beginners a lot, as long as it uses words you know. Look at step 3 of the recipe again.",
            "ask": "That keeps the explanation within what you know, and you know where the word comes back.",
        },
    )
    choice_new_word
    return (choice_new_word,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Using AI tools with your eyes open](/course/open/start-using-ai-tools?lang=en) you learned to check what an assistant says; here you use that while learning. In [How a lesson works](/course/open/start-how-this-course-works?lang=en) you met the checks and the hints; your assistant comes on top of those, not instead of them. From the next lesson on you write Python, and that is where an assistant that gives hints instead of answers helps you the most.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_new_word, choice_trust_check, result_ask):
    L.wrap_up(result_ask, choice_trust_check, choice_new_word)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
