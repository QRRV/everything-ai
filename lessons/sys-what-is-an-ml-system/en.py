# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-what-is-an-ml-system. Route: course/route/lessen/systemen.yaml. Checks, the course's spam service and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-what-is-an-ml-system", "en")
    C = L.checks   # the course's spam service, its parts and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [The machine learning workflow](/course/open/ml-workflow?lang=en) and [Every step in every fold: pipelines](/course/open/ml-pipelines?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_test():
    quick_test = L.choice(
        "quick-test",
        question="In the workflow of a project, when do you measure the model on the test set?",
        options={
            "rounds": "After every round of the workflow, to see whether the model got better",
            "start": "Before training, so that you know the goal",
            "right": "Once, at the end, after choosing on the validation set",
            "never": "Never: the validation set says enough",
        },
        correct="right",
        why={
            "right": "The test set stays out of every choice, so that it measures how the model does on data it has never seen. Once a model is in use, every new message is such data, and it keeps coming: this lesson is about what a model needs around it then.",
            "rounds": "If the test set helps choose after every round, the choices fit the test set too. Which set do the rounds choose on? See [The machine learning workflow](/course/open/ml-workflow?lang=en).",
            "start": "The goal is agreed on before training, but from what each mistake costs, not from the test set. When does the test set come in? See [The machine learning workflow](/course/open/ml-workflow?lang=en).",
            "never": "The validation set has helped choose the model, so its score is a little too good. Which set measures the chosen model fairly, and how often? See [The machine learning workflow](/course/open/ml-workflow?lang=en).",
        },
    )
    quick_test
    return (quick_test,)


@app.cell(hide_code=True)
def quick_check_pipeline():
    quick_pipe = L.choice(
        "quick-pipe",
        question="A pipeline of `StandardScaler()` and a model is fitted on the training set. What does `pipe.predict` do with a new example?",
        options={
            "learns": "It first learns a new mean and spread from the one new example alone",
            "right": "It scales the example with what the training set gave, and predicts",
            "raw": "It predicts on the raw example: scaling is for training only",
            "refit": "It fits the whole pipeline again, with the new example in it",
        },
        correct="right",
        why={
            "right": "`pipe.predict` only changes new examples with what each step learned from the training set. In a running system, every new message has to go through exactly the steps the model was trained with: a step that works differently shows the model data it never learned from.",
            "learns": "`fit` learns, and `predict` only uses what was learned. What would one new example give as its own mean and spread? See [Every step in every fold: pipelines](/course/open/ml-pipelines?lang=en).",
            "raw": "The model learned from scaled examples. What happens to its predictions on numbers of another size? See [Every step in every fold: pipelines](/course/open/ml-pipelines?lang=en).",
            "refit": "`pipe.fit` learns every step, and `pipe.predict` uses them. Which of the two is called here? See [Every step in every fold: pipelines](/course/open/ml-pipelines?lang=en).",
        },
    )
    quick_pipe
    return (quick_pipe,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In 2006, Netflix offered a million dollars to whoever could predict how its members rate movies 10% better than its own system. In 2009, a team won, with a combination of many models. Netflix never used the winning model. In 2012 it explained why. A year into the competition, a team had won a first, smaller prize with a combination of 107 algorithms; Netflix took the two best of them, built them into its service and kept them running there. The extra accuracy of the final winner "did not seem to justify the engineering effort" of bringing it into that service, and by then the attention of Netflix had moved elsewhere. A model with the best score on a test set is not yet something people can use: it has to get its data, answer quickly, keep working when the data changes, and be looked after for years. This lesson is about everything around the model.

    **An everyday example.** A good recipe does not make a restaurant. Around the recipe, a restaurant buys ingredients and checks that they are fresh, takes orders, serves every table within a reasonable time, listens to complaints and changes the menu with the seasons. The recipe is one page; running the restaurant is all the rest. And a dish that takes three hours does not work on a busy evening, however good it tastes.
    """)
    return


@app.cell(hide_code=True)
def predict_share():
    guess_share = L.guess(
        "share",
        question="The course wrote a small spam service in Python, in eight parts, with a model from scikit-learn. How much of the code of the service is the model, do you think?",
        options={
            "most": "Most of it: the model does the real work",
            "half": "About half",
            "quarter": "About a quarter",
            "right": "A few percent",
        },
        reveal="**A few percent.** Of the 118 lines of code of the service, 5 are the model: 4.2%. The learning itself is done by scikit-learn, and the five lines hand it the data. All the other lines take messages in, check them, turn them into numbers, give answers, watch the answers and train the model again.",
    )
    guess_share
    return (guess_share,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A model is part of a system.** In the projects so far, the data was ready in a file, you trained a model once and measured it on a test set. A model that people use needs much more around it. An **ML system** is the model together with everything that makes it useful in practice: the data coming in, checks on that data, features, training, serving the answers, watching how it goes, and the people who look after it all.

        **The parts of the course's spam service.** The course wrote a small ML system: a service that gets text messages and answers spam, ok, or no score. Each message goes down through five parts. *Collection* takes each message once, *verification* checks that the message is usable, *features* turns its words into counts, the *model* gives a probability of spam, and *serving* turns that into an answer: always, also when the model cannot help. Beside them, *settings* holds the numbers the other parts read, such as the threshold, *monitoring* watches the answers, and *feedback* keeps the messages whose answer users correct, to train the model again.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.accordion({"The code of the service, part by part": mo.vstack([
            mo.vstack([mo.md(f"**{_name}**"), mo.md("```python\n" + _code + "\n```")])
            for _name, _code in C.parts_accordion(L.lang).items()
        ])}),
        mo.md(r"""
        **The model is a small part.** In 2015, D. Sculley and others at Google wrote *Hidden Technical Debt in Machine Learning Systems*. Its best-known figure draws a real ML system as boxes: configuration, data collection, data verification, feature extraction, serving, monitoring, tools for analysis, tools that manage computers and processes, and in the middle one small box: the ML code. The figure is a sketch, not a measurement, but people who build ML systems recognize it. In the course's service, the model is short because scikit-learn does the learning: `LogisticRegression().fit(rows, labels)` is one line. That is usual: the model often comes from a library, and the code that connects it to everything else is the system's own. The paper calls the code that only gets data into and out of such a general package *glue code*.

        **Technical debt.** A shortcut in code works like a loan: it saves time now and costs more later, every time someone has to work around it. Ward Cunningham compared this to a debt in 1992; since then, it has been called **technical debt**. Sculley and others show that ML systems have their own kinds of it, on top of the usual ones. One is *CACE*: changing anything changes everything. A model mixes all its features together: change, add or remove one, and after training again the model uses all the others differently, so no change stands on its own. A related problem: a model depends on its features as they were made during training; change how one feature is made, and the model answers differently, without one line of its own code changing. Another is the *pipeline jungle*: preparation of data that grows one fix at a time, until nobody dares to change it. And a model can steer its own future data: what it decides now ends up in what it learns from later.

        **Research and production.** A model in research and the same model in production, in use by real people, are asked for different things:

        | | research | production |
        |---|---|---|
        | the goal | the best score on a fixed test set | good answers for real users, quick and cheap enough, for years |
        | the data | collected once, cleaned, fixed | keeps coming in, changes, and is sometimes broken |
        | time | how long training takes | how long every single answer takes, for many users at once |
        | done | when the results are written down | only when the system is switched off |

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **ML system** | | a model together with everything around it that makes it useful in practice: the data coming in, checks, features, training, serving, monitoring and the people who look after it | |
        | **technical debt** | | the cost later of shortcuts taken now, which make a system harder to change; ML systems add their own kinds, such as CACE | |
        """),
        L.watch_out("A better score on the test set does not make a better system. A model that is a little more accurate but ten times as slow, or much harder to keep running, can make the whole system worse."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Looking at an ML system:

    1. Follow one input from the moment it comes in to the answer, and name each part it passes.
    2. Name the parts beside that path: settings, monitoring, feedback and training again.
    3. For each part, ask what real data can do to it: come in twice, be empty, be too long, change over time.
    4. Count how much of the code is the model, and how much is everything around it.
    5. Write down what production asks: how quickly an answer must come, for how many users, at what cost. Then choose the model that fits, not just the one with the best score.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.part_code("model", L.lang)` gives the code of the model's part of the service as one text. The cell shows that text, cuts it into lines with `code.split("\n")`, and counts all of them. Run the cell.
    """)
    return


@app.cell
def example_lines():
    code = C.part_code("model", L.lang)
    print(code)
    lines = code.split("\n")
    print()
    print("all lines:", len(lines))
    return code, lines


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The part has 11 lines, but 4 of them are empty and 2 are comments, which explain the code and do nothing. Only 5 lines do work: the `import`, two `def` lines and two `return` lines. Exercise 1 counts only such lines of code.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lines():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Count the lines of code

        `code_lines(code)` gets the code of a part as one text, and gives back its number of lines of code: lines that are not empty and not a comment. `code.split("\n")` cuts the text into lines, and `text` is the `line` without the spaces at its start and end, from `line.strip()`. For a line of code, `count` goes up by one. Replace the `...` with the condition: `text` is not empty, and its first character is not `#`. Python's `and` checks its left side first, and looks at its right side only when the left side is `True`.

        The check calls `code_lines(C.part_code("model"))`, which should give 5. Then the seven other parts of the service, a short piece of code with an indented comment and a line of only spaces, an empty text, and a text of only comments.
        """),
        L.task("Count only the lines that are not empty and not a comment."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lines():
    box_lines = L.code_box("count-lines", lines=8)
    box_lines
    return (box_lines,)


@app.cell(hide_code=True)
def check_lines(box_lines):
    result_lines = L.check("count-lines", box_lines.value)
    result_lines
    return (result_lines,)


@app.cell(hide_code=True)
def hints_lines():
    L.hints(
        "count-lines",
        [
            "Step 4 of the recipe. The condition has two parts joined by `and`: `text` is not empty, and the first character of `text` is not `#`.",
            "`text != \"\"` is the first part. The first character of a text is the one at place 0. `and` stops at a left side that is `False`, so on an empty line Python never looks for a first character that is not there.",
            "`        if text != \"\" and ___:`",
        ],
        why="`if text != \"\" and text[0] != \"#\":` lets through a line with something on it that does not begin with `#`. The spaces are gone from `text`, so a comment inside a function begins with `#` too.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_budget():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The best model within the time

        A team compares six spam models: the accuracy on a validation set of many messages, and the milliseconds one message takes. The numbers are an example the course made up. The mail server takes in 100 messages per second on one core, so one message may take at most 10 ms.

        | model | accuracy | ms per message |
        |---|---|---|
        | word list | 0.80 | 0.002 |
        | naive Bayes | 0.93 | 0.02 |
        | logistic regression | 0.95 | 0.05 |
        | small network | 0.96 | 1.5 |
        | 30 models together | 0.97 | 40 |
        | large language model | 0.98 | 800 |

        `best_in_budget(models, budget)` gets a list of models, each a tuple `(name, accuracy, ms)`, and a budget in milliseconds per message. It gives back the name of the most accurate model that fits within the budget, or `None` when no model fits. The loop `for name, accuracy, ms in models` goes through the models; `best_name` and `best_accuracy` hold the best model so far. Replace the `...` with the condition: the model fits within `budget`, and its `accuracy` is higher than `best_accuracy`. In *Let the metric choose* in [The machine learning workflow](/course/open/ml-workflow?lang=en), a precision of at least 0.9 was the condition; here, time is.

        The check calls `best_in_budget(C.MODELS, 10.0)`: the six models of the table, with a budget of 10 ms. Then the same models with budgets of 1,000, 1.5 and 0.001 ms, an empty list, and three other models.
        """),
        L.task("Give back the most accurate model whose time per message fits within the budget."),
    ])
    return


@app.cell(hide_code=True)
def exercise_budget():
    box_budget = L.code_box("budget", lines=9)
    box_budget
    return (box_budget,)


@app.cell(hide_code=True)
def check_budget(box_budget):
    result_budget = L.check("budget", box_budget.value)
    result_budget
    return (result_budget,)


@app.cell(hide_code=True)
def hints_budget():
    L.hints(
        "budget",
        [
            "Step 5 of the recipe. The condition has two parts joined by `and`: the second, `accuracy > best_accuracy`, compares with the best model so far, and the first compares `ms` with `budget`.",
            "A model fits when one message takes it no longer than the budget: with a budget of 1.5 ms, the small network of exactly 1.5 ms still fits.",
            "`        if ___ and accuracy > best_accuracy:`",
        ],
        why="`if ms <= budget and accuracy > best_accuracy:` skips every model that is too slow, and of the others keeps the most accurate. With 10 ms, the 30 models together and the large language model are too slow, and the small network is the best of the rest.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### The service at work

    The demo sends ten messages through the course's service, one at a time, as they might come in: with capitals and punctuation, one that comes in twice, and some that cannot be scored. For each message, the picture shows what each part did with it; a part that did not run is dashed, and a bar on the left marks a part that stopped the message, raised an alert or got a user's mark. In message 5, a user moves the message out of the spam folder.
    """)
    return


@app.cell(hide_code=True)
def demo_service():
    L.explore("message", "message", lambda _v: C.frame(int(_v), L.lang), start=1, values=list(range(1, 11)), role="",
              task="Go through the ten messages: for how many of them does the model give the answer?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    The model gives the answer for 5 of the 10 messages. One message came in twice, three are not usable (empty, hardly any letters, too long), and one has no word the model knows: the other parts handle all of these without the model. In message 5, the model was wrong, and only a user noticed: the feedback keeps the message to train on later. At messages 8 and 9, the monitoring raises an alert because most of the latest answers have no score, and at message 10 because all the scored ones are spam. None of this shows up in a score on a test set.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cace():
    mo.md(r"""
    ### Exercise 3. A small change

    The team changes one line of the features: `text.lower()` becomes `text`, so from now on words keep their capitals. The model is not trained again.
    """)
    return


@app.cell(hide_code=True)
def choice_cace_cell():
    choice_cace = L.choice(
        "cace",
        question="What happens to the answers of the service?",
        options={
            "nothing": "Nothing: not one line of the model changed",
            "right": "Some answers change, though the model's code stays the same",
            "error": "The service stops with an error",
            "learns": "The model learns the new words by itself, while it is answering",
        },
        correct="right",
        why={
            "right": "The model learned from the features as they were made during training, and now gets other counts for the same messages. Sculley and others call this an unstable data dependency: something the model depends on changed after training. In a pipeline, as in [Every step in every fold: pipelines](/course/open/ml-pipelines?lang=en), the steps and the model are fitted together, so fitting again trains the model on the new features; change a step without fitting again, and nothing warns you there either.",
            "nothing": "The model gets its counts from the features. What does it get for *WIN* after the change, when its vocabulary only holds *win*?",
            "error": "The count rows are as long as before, with a number for every word of the vocabulary. Is there anything that would stop the code?",
            "learns": "The model only learns when `train` is called, in the feedback. What happens between two trainings?",
        },
    )
    choice_cace
    return (choice_cace,)


@app.cell(hide_code=True)
def text_exercise_loop():
    mo.md(r"""
    ### Exercise 4. Learning from the users

    The service is trained again every week, on the messages users marked: spam they found among their mail, and ok messages they moved out of the spam folder. Few users ever look in their spam folder.
    """)
    return


@app.cell(hide_code=True)
def choice_loop_cell():
    choice_loop = L.choice(
        "loop",
        question="What can happen to the model over the weeks?",
        options={
            "better": "It only gets better, since it gets more data every week",
            "right": "It rarely learns from its mistakes on ok messages",
            "same": "Nothing: users mark their messages right",
            "forget": "It forgets all spam from before",
        },
        correct="right",
        why={
            "right": "An ok message the model puts in the spam folder is rarely seen, so it is rarely marked: what the model decides now steers what it learns from later. Sculley and others call this a direct feedback loop. A team can check a sample of the spam folder by hand, or let a few messages through unfiltered.",
            "better": "More data only helps when it shows the mistakes. Which mistakes do users see, and which do they hardly ever see?",
            "same": "The marks that come in may be right. But which messages never get a mark?",
            "forget": "The old examples stay in what the model is trained on. Which mistakes go missing from the new marks?",
        },
    )
    choice_loop
    return (choice_loop,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [The machine learning workflow](/course/open/ml-workflow?lang=en), a project ended with one measurement on the test set, and in [Every step in every fold: pipelines](/course/open/ml-pipelines?lang=en) the steps and the model became one whole. This lesson put that whole into a system: parts that take in data, check it, answer, watch and learn again, with the model as a small part in the middle. In the next lesson, *From idea to retirement: the life cycle of an AI system*, you follow such a system through time, from the first idea to the day it is switched off, and see who is involved at each stage.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_cace, choice_loop, result_budget, result_lines):
    L.wrap_up(result_lines, result_budget, choice_cace, choice_loop)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
