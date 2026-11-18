# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-rules-and-dilemmas. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-rules-and-dilemmas", "en")
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
def quick_check_consequences():
    quick_consequences = L.choice(
        "quick-consequences",
        question="Which question does consequentialism ask about a choice?",
        options={
            "duty": "Whether it keeps a duty or a right, whatever the outcome turns out to be",
            "right": "What good and harm it brings about, added up over everyone",
            "character": "What a person of good character would do in that same place",
        },
        correct="right",
        why={
            "duty": "That is the question of deontology, which holds on to duties and rights whatever the outcome. See *The idea* in [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).",
            "right": "Consequentialism judges a choice by what it brings about: the best choice does the most good, added up over everyone.",
            "character": "That is the question of virtue ethics. Which theory adds up outcomes? See *The idea* in [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).",
        },
    )
    quick_consequences
    return (quick_consequences,)


@app.cell(hide_code=True)
def quick_check_care():
    quick_care = L.choice(
        "quick-care",
        question="One theory starts from relationships and from the people who depend on others. Which one, and what does it ask?",
        options={
            "virtue": "Virtue ethics: what would a person of good character do?",
            "contract": "A fair agreement: which rule could everyone accept?",
            "right": "Care ethics: who depends on whom, and what do they need?",
        },
        correct="right",
        why={
            "virtue": "Virtue ethics asks about character, not about who depends on whom. See *The idea* in [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).",
            "contract": "A fair agreement asks which rule you would choose without knowing who you will be. See *The idea* in [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).",
            "right": "Care ethics starts from relationships and from those who need others, and listens to what they say they need.",
        },
    )
    quick_care
    return (quick_care,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    People have long wanted rules that make machines behave well. In 1942, Isaac Asimov wrote three laws for the robots in his stories, and many of those stories are about how the laws go wrong. Today the question is no longer only fiction: a self-driving car that cannot stop in time has to do something, and someone has to decide beforehand what. Rules are easy to write down. This lesson writes some as code, to see where they hold and where they break.

    **An everyday example.** A family has two house rules: no phones at the dinner table, and always answer when grandma calls. Both are fine, until grandma calls during dinner.
    """)
    return


@app.cell(hide_code=True)
def predict_decisions():
    guess_decisions = L.guess(
        "decisions",
        question="A website called the Moral Machine asked visitors what a self-driving car should do in a crash it cannot avoid, such as hitting three people or swerving into a wall. How many decisions do you think it collected?",
        options={
            "thousands": "About 40,000",
            "millions": "About 4 million",
            "right": "About 40 million",
        },
        reveal="**About 40 million**, in ten languages, from millions of people in 233 countries and territories (Awad and others, 2018). Some preferences held almost everywhere; others differed between regions of the world, as *The idea* shows.",
    )
    guess_decisions
    return (guess_decisions,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Machine ethics** asks how a machine can act in a way that is morally right: by following rules written by people, by learning from examples, or by a mix of both. Asimov's **Three Laws of Robotics** are the best-known attempt with rules. In short:

        1. A robot may not harm a person, or let a person come to harm by doing nothing.
        2. A robot must obey people, unless that breaks the first law.
        3. A robot must protect itself, unless that breaks the first or the second law.

        As code, each law is a filter on the actions a robot could take, applied in order. That works as long as some action passes. But the laws do not say what *harm* is, or whose order counts when two people disagree, and when every action harms someone, the first law forbids them all.

        That last case is the *trolley problem*, a thought experiment of the philosopher Philippa Foot from 1967: a runaway tram will kill five people unless its driver steers it onto a track where it kills one. The **Moral Machine** put such choices to millions of people, for self-driving cars. Over all the answers, three preferences were strongest: sparing people over animals, more lives over fewer, and the young over the old. But in a group of countries the authors called the Eastern cluster, the preference for sparing the young was much weaker. One fixed rule goes against what many people would choose, and a majority vote is not yet an ethical argument.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **machine ethics** | the part of AI ethics that tries to build moral reasoning into machines themselves, from rules, from examples, or both |
        | **Three Laws of Robotics** | Isaac Asimov's made-up rules for robots: do not harm people, obey people, protect yourself, in that order |
        | **Moral Machine** | an online experiment in which millions of people chose what a self-driving car should do in a crash it cannot avoid; preferences differed between regions |
        """),
        L.watch_out("The trolley problem is a thought experiment, not a picture of how cars crash: a real car brakes and steers without knowing for sure what will happen. Its use is to show where rules clash, not to design a car."),
        mo.Html(C.picture({"title": "Actions still allowed after each law", "count": "actions allowed",
                           "stages": ["start", "law 1", "law 2", "law 3"], "kitchen": "robot bringing the post", "car": "car that cannot stop",
                           "alt": "For a robot bringing the post, 4 actions at the start, 3 after the first law, 2 after the second and 1 after the third. For a car that cannot stop, 2 actions at the start and 0 after the first law."})),
        mo.md(r"""
        Two situations, made up by the course. Filled bars: a robot asked to bring the post has 4 actions, and each law takes some away until 1 is left. Open bars: a car that cannot stop has 2, and both harm someone, so after the first law nothing is left.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To test a rule for a machine:

    1. Write the rule as code: which actions does it keep?
    2. Try it on an easy case, then on a case where every action costs something.
    3. Look for words the rule does not define, such as harm.
    4. Ask what the rule does when two people want opposite things.
    5. Where the rule says nothing, write down who should decide instead, and how.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The robot of the picture is asked to bring the post. Three dictionaries say, for each action, how many people it harms, whether it does what it was ordered, and whether the robot stays whole. Each law keeps the actions that pass it, out of those the law before it kept. Run the cell.
    """)
    return


@app.cell
def example_kitchen():
    _harmed = {"walk around the child": 0, "push past the child": 1, "stay in the corner": 0, "jump down the stairs": 0}
    _ordered = {"walk around the child": True, "push past the child": True, "stay in the corner": False, "jump down the stairs": True}
    _whole = {"walk around the child": True, "push past the child": True, "stay in the corner": True, "jump down the stairs": False}

    # 1. law 1: harm no one
    _law1 = []
    for _action in _harmed:
        if _harmed[_action] == 0:
            _law1.append(_action)
    print("law 1:", _law1)
    # law 2: obey, among what law 1 allows
    _law2 = []
    for _action in _law1:
        if _ordered[_action]:
            _law2.append(_action)
    print("law 2:", _law2)
    # law 3: stay whole, among what law 2 allows
    _law3 = []
    for _action in _law2:
        if _whole[_action]:
            _law3.append(_action)
    print("law 3:", _law3)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first law takes away pushing past the child, the second staying in the corner, the third jumping down the stairs: the robot walks around the child. Step 2 of the recipe asks for a harder case: Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_allowed():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The car that cannot stop

        A self-driving car cannot stop in time. If it keeps going, it hits three people; if it swerves into the wall, it hurts its one passenger. `harmed` says how many people each action harms, and the loop goes through the actions with `action`. Replace the `...` with the first law, as in the worked example. The last line is `allowed`, so the check sees it. The check also tries another situation.
        """),
        L.task("Put in `allowed` the names of the actions that harm no one."),
    ])
    return


@app.cell(hide_code=True)
def exercise_allowed():
    box_allowed = L.code_box("allowed", lines=7)
    box_allowed
    return (box_allowed,)


@app.cell(hide_code=True)
def check_allowed(box_allowed):
    result_allowed = L.check("allowed", box_allowed.value)
    result_allowed
    return (result_allowed,)


@app.cell(hide_code=True)
def hints_allowed():
    L.hints(
        "allowed",
        [
            "Step 1 of the recipe, as the first law in the worked example: two lines inside the loop.",
            "Which test says that an action harms no one? And what does the loop do with an action that passes?",
            "`if harmed[action] == ___:` and under it `allowed.___(action)`.",
        ],
        why="`if harmed[action] == 0:` with `allowed.append(action)` under it keeps the actions that harm no one. Here that is none of them: `[]`. The first law forbids every action, and does not say which harm is less.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_gaps():
    mo.md(r"""
    ### Exercise 2. Settled or not?

    Six cases for a robot with the three laws. Put each one with what the laws do with it.
    """)
    return


@app.cell(hide_code=True)
def sort_gaps_cell():
    sort_gaps = L.sort("gaps")
    sort_gaps
    return (sort_gaps,)


@app.cell(hide_code=True)
def check_gaps(sort_gaps):
    result_gaps = L.check("gaps", sort_gaps)
    result_gaps
    return (result_gaps,)


@app.cell(hide_code=True)
def hints_gaps():
    L.hints(
        "gaps",
        [
            "Ask for each case: does one law clearly come before the other, or do the laws leave the answer open?",
            "In three cases, two laws pull different ways and their order decides. In which three is there no order that helps?",
            "Settled: the order to hit, \\_\\_\\_ and \\_\\_\\_. No answer: every way harms, \\_\\_\\_ and \\_\\_\\_.",
        ],
        why="**Settled:** the order to hit (the first law comes before the second), into the fire (the second comes before the third) and the falling wall (only the third applies). **No answer:** every way harms (the first law forbids all), a hard truth (harm is not defined) and two orders (the second law does not say whose).",
    )
    return


@app.cell(hide_code=True)
def text_exercise_moral_machine():
    mo.md(r"""
    ### Exercise 3. One rule for everyone?
    """)
    return


@app.cell(hide_code=True)
def choice_moral_machine_cell():
    choice_moral_machine = L.choice(
        "moral-machine",
        question="The Moral Machine found that some preferences differ between regions of the world. A car maker wants one rule for all its cars. What follows from the results?",
        options={
            "majority": "Take the choice of the world's majority; then the rule is right for everyone",
            "right": "Any fixed rule goes against what many people would choose",
            "small": "Nothing: the differences were too small to matter",
        },
        correct="right",
        why={
            "majority": "A majority shows what most people chose, not what is right, and the people in the minority still have to live with the rule. What do the regional differences mean for one rule? See *The idea*.",
            "right": "Where preferences differ, one rule always goes against some of them. Who decides, and how, is the ethical question, not something the data settles.",
            "small": "Some differences were large, such as how much people in different regions wanted to spare the young. See *The idea*.",
        },
    )
    choice_moral_machine
    return (choice_moral_machine,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en) you judged one choice through several theories. Asimov's laws are a set of duties in a fixed order, deontology written as code, and a dilemma where every action harms someone is where such duties run out. In the next lesson, *Principles and guidelines*, you look at the principles governments and companies wrote for AI instead of rules: short words that almost everyone agrees on.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_moral_machine, result_allowed, result_gaps):
    L.wrap_up(result_allowed, result_gaps, choice_moral_machine)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
