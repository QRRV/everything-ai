# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-wellbeing. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-wellbeing", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rule():
    quick_rule = L.choice(
        "quick-rule",
        question="A rule says that a model may never use someone's religion, even where that would make its predictions better. Which theory does that rule fit best?",
        options={
            "outcome": "Consequentialism: it gives the best outcome overall",
            "right": "Deontology: some rules hold whatever the outcome",
            "care": "Care ethics: it listens to the people involved",
        },
        correct="right",
        why={
            "outcome": "The rule holds even where breaking it would give better predictions. Which theory keeps a rule whatever the outcome? See *The idea* in [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).",
            "right": "The rule holds even when breaking it would do better, which is what a duty or a right does.",
            "care": "Nobody is asked anything here: it is a rule that holds whatever happens. Which theory does that? See *The idea* in [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).",
        },
    )
    quick_rule
    return (quick_rule,)


@app.cell(hide_code=True)
def quick_check_veil():
    quick_veil = L.choice(
        "quick-veil",
        question="Behind Rawls's veil of ignorance, read as in that lesson, which of two policies do you choose?",
        options={
            "right": "The one under which the worst-off group is best off",
            "total": "The one that does the most good in total",
            "vote": "The one that most of the people in the room would vote for",
        },
        correct="right",
        why={
            "right": "You do not know in which group you will end up, so you look after the group that is worst off.",
            "total": "That is what consequentialism adds up. Behind the veil you do not know who you will be: which group do you look at? See Exercise 1 in [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).",
            "vote": "Behind the veil nobody knows who they will be, so a vote tells you nothing new. Which group do you look at? See Exercise 1 in [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).",
        },
    )
    quick_veil
    return (quick_veil,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    An app that picks which videos to show you learns from what you click, and shows more of it. That works: people click. But a system that optimizes a number has made a choice about what is good for people, often without saying so. In 2018, researchers paid people in the US to stop using Facebook for four weeks, and compared them with people who kept using it.

    **An everyday example.** A shop that puts sweets next to the till sells more sweets. That is what customers pick in the moment, not what most of them would choose for themselves.
    """)
    return


@app.cell(hide_code=True)
def predict_facebook():
    guess_facebook = L.guess(
        "facebook",
        question="How do you think the people who stopped using Facebook for four weeks felt, on average, compared with those who kept using it?",
        options={
            "worse": "Worse: they missed their friends and the news",
            "same": "No different",
            "right": "A little better",
        },
        reveal="**A little better.** On average they said they were a little happier and more satisfied with their lives, and in the weeks after, they used Facebook less than the people who had kept using it. They also knew a little less about the news. What people click on is not always what makes their lives go better.",
    )
    guess_facebook
    return (guess_facebook,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Well-being** is what makes a life go well for the person living it. Philosophers give different answers to what that is, and each answer measures something else:

        - *Hedonism*: well-being is pleasure and the absence of pain. You measure feelings.
        - *Preference satisfaction*: well-being is getting what you want. A click is a want in the moment; what you would want with time to think can be different.
        - *A list of goods*: some things are good for anyone, wanted or not, such as health, friendship and knowledge.
        - The **capability approach** of Amartya Sen and Martha Nussbaum: look at what people are really able to do and be, such as being able to read, to get around, and to take part in their community. Not what they have or feel, but their real freedom to choose a life. A list counts the goods themselves; the capability approach asks whether people are free to reach them.

        Whatever a system optimizes, it has picked one of these. Clicks are the preference of the moment. The consequentialism of the previous lesson adds up well-being, but it has to say which well-being it adds up.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **well-being** | what makes a life go well for the person living it; theories say pleasure, getting what you want, or a list of goods such as health and friendship |
        | **capability approach** | judging well-being by what people are really able to do and be, such as reading, getting around and taking part in their community (Sen and Nussbaum) |
        """),
        L.watch_out("The capability approach does not mean that the app knows best. It asks what people are able to do, and leaves the choice of what to do with it to them."),
        mo.Html(C.picture({"title": "A feed ranked by clicks", "share": "share of viewers", "clicks": "click", "learned": "could do something new",
                           "titles": ["prank", "gossip", "unboxing", "fix a bike", "cook rice", "first aid"],
                           "alt": "Six videos in the order an app shows them. The share who click falls from 0.62 to 0.12 along the feed; the share who could do something new afterwards rises from 0 to 0.7."})),
        mo.md(r"""
        The feed of this lesson, made up by the course: six videos in the order the app shows them, ranked by the share of viewers who click. Open bars are the share who click, filled bars the share who could do something new afterwards, such as fix a bike: they go the other way.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To see what a system makes better:

    1. Write down what the system optimizes now, such as clicks, minutes or stars.
    2. Ask which theory of well-being that measure belongs to.
    3. Pick a measure for the other theories too: a feeling, a want on second thought, a good on the list, something people can do.
    4. Score what people actually get, such as the top of a feed, on every measure.
    5. Where the scores differ, write down which one the system serves, and which it leaves out.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The feed of the picture, in the app's order. Most viewers only see the top of a feed, so step 4 scores the first three rows: `feed.iloc[:3]` keeps them, as a slice of a list keeps its first three values. Run the cell.
    """)
    return


@app.cell
def example_feed():
    import pandas as pd

    _titles = ["prank", "gossip", "unboxing", "fix a bike", "cook rice", "first aid"]
    _clicks = [0.62, 0.55, 0.48, 0.21, 0.18, 0.12]
    _learned = [0.0, 0.0, 0.1, 0.6, 0.5, 0.7]
    _feed = pd.DataFrame({"title": _titles, "clicks": _clicks, "learned": _learned})
    # 1. what the app optimizes: clicks, in the order it shows the videos
    print(_feed)
    # 4. what people get: the top three, and the whole feed for comparison
    print(round(_feed.iloc[:3]["clicks"].mean(), 2))
    print(round(_feed["clicks"].mean(), 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The top three get a mean of 0.55 for clicks, against 0.36 for the whole feed: the app does what it was built for. Step 2: a click is a preference of the moment. Whether viewers are better off, only the other measures can say. Exercise 1 scores the same top three on them.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_scores():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Three scores for one feed

        The same feed, now with `glad` as well: the share of viewers who say a week later that they were glad they watched. That is a want on second thought. The numbers are made up by the course. Replace the `...` with one line, as in the worked example. The last line is `scores`, so the check sees it. The check also tries other numbers for `learned`.
        """),
        L.task("Put in `scores` the mean of each measure over the first three videos of the feed, under the name of the measure."),
    ])
    return


@app.cell(hide_code=True)
def exercise_scores():
    box_scores = L.code_box("scores", lines=13)
    box_scores
    return (box_scores,)


@app.cell(hide_code=True)
def check_scores(box_scores):
    result_scores = L.check("scores", box_scores.value)
    result_scores
    return (result_scores,)


@app.cell(hide_code=True)
def hints_scores():
    L.hints(
        "scores",
        [
            "Step 4 of the recipe, as in the worked example, but with the column whose name is in `measure`.",
            "Which rows does a viewer see first, and how do you keep only those? Then take the mean of one column.",
            "`scores[measure] = feed.___[___][measure].mean()`.",
        ],
        why="`feed.iloc[:3]` keeps the first three videos, and `[measure]` the column of this round: about 0.55 for clicks, 0.3 for glad and 0.03 for learned. The feed that wins on clicks loses on what viewers are glad about later and on what they can do afterwards.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_measures():
    mo.md(r"""
    ### Exercise 2. Which theory?

    Eight things an app could measure. Put each one with the theory of well-being it belongs to.
    """)
    return


@app.cell(hide_code=True)
def sort_measures_cell():
    sort_measures = L.sort("measures")
    sort_measures
    return (sort_measures,)


@app.cell(hide_code=True)
def check_measures(sort_measures):
    result_measures = L.check("measures", sort_measures)
    result_measures
    return (result_measures,)


@app.cell(hide_code=True)
def hints_measures():
    L.hints(
        "measures",
        [
            "Ask for each measure: is it a feeling, a want, a good that counts whether you want it or not, or something a person is able to do?",
            "*Health and friends* says *whatever they want*. Which two measures ask what people are able to do?",
            "Pleasure: smiles and \\_\\_\\_. Getting what you want: clicks and \\_\\_\\_. A list: health and friends, and \\_\\_\\_. What people can do: a new skill and \\_\\_\\_.",
        ],
        why="**Pleasure:** smiles and mood. **Getting what you want:** clicks and asked for. **A list of goods:** health and friends, and knowledge. **What people can do and be:** a new skill and taking part.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_goal():
    mo.md(r"""
    ### Exercise 3. A goal for an app
    """)
    return


@app.cell(hide_code=True)
def choice_goal_cell():
    choice_goal = L.choice(
        "goal",
        question="A language app is built to make one number go up. Which number fits the capability approach best?",
        options={
            "minutes": "Minutes people spend in the app each day",
            "stars": "The stars people give it in the app store after a week",
            "right": "How many can hold a short conversation after a month",
        },
        correct="right",
        why={
            "minutes": "Minutes in the app are what people pick in the moment, and an app can keep people busy without teaching them anything. What are they able to do afterwards? See *The idea*.",
            "stars": "Stars say how much people liked the app, a feeling or a want. What are they able to do afterwards? See *The idea*.",
            "right": "Holding a conversation is something people are now able to do, a real freedom they did not have before. What they do with it is up to them.",
        },
    )
    choice_goal
    return (choice_goal,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en) you judged one choice through several theories; here you saw that even "the most good" needs a theory of what is good for people. Scoring what people actually get on more than one measure is the same habit as counting per group in [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en): one number can hide what matters. In the next lesson, *Rules and dilemmas for machines*, you try to write rules for a machine, and see where they clash.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_goal, result_measures, result_scores):
    L.wrap_up(result_scores, result_measures, choice_goal)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
