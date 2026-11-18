# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-professional-ethics. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-professional-ethics", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Principles and guidelines](/course/open/rai-principles?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_common():
    quick_common = L.choice(
        "quick-common",
        question="Jobin, Ienca and Vayena read 84 documents with principles for AI. Which principle did most of them name?",
        options={
            "right": "Transparency",
            "solidarity": "Solidarity",
            "freedom": "Freedom and autonomy",
        },
        correct="right",
        why={
            "right": "73 of the 84 documents name transparency, more than any other principle.",
            "solidarity": "Solidarity came last: only 6 of the 84 documents name it. See the picture in [Principles and guidelines](/course/open/rai-principles?lang=en).",
            "freedom": "34 of the 84 documents name freedom and autonomy, fewer than half. See the picture in [Principles and guidelines](/course/open/rai-principles?lang=en).",
        },
    )
    quick_common
    return (quick_common,)


@app.cell(hide_code=True)
def quick_check_same_word():
    quick_same_word = L.choice(
        "quick-same-word",
        question="Two documents both name fairness as a principle. What does that tell you?",
        options={
            "same": "That they ask for exactly the same things",
            "right": "Little; read what each one means by it",
            "source": "That both come from the same organization",
        },
        correct="right",
        why={
            "same": "The same word can ask for different things, and two documents can even clash. See the *Watch out* box in [Principles and guidelines](/course/open/rai-principles?lang=en).",
            "right": "The shared word hides different meanings: to know what a principle asks for, read what it means in that document.",
            "source": "The 84 documents came from governments, companies and others. Two documents can share a word and still ask for different things. See *The idea* in [Principles and guidelines](/course/open/rai-principles?lang=en).",
        },
    )
    quick_same_word
    return (quick_same_word,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Principles are written by organizations. But the person who builds the system is often the first to see that it can hurt someone, and the one who is asked to ship it anyway. Professions write down what their members owe the public, so that a person in that spot has something to point to. And even a choice that looks purely technical, such as which setting of a model to use, rests on values. This lesson is about both.

    **An everyday example.** A doctor promises to put the patient first, whoever pays the bill. And how sure a doctor wants to be before calling a spot on a scan harmless depends on how bad it would be to miss an illness.
    """)
    return


@app.cell(hide_code=True)
def predict_need():
    guess_need = L.guess(
        "need",
        question="Researchers read 100 highly cited machine learning papers from two of the biggest conferences. In how many do you think the paper explained how the work met a need in society?",
        options={
            "most": "About 75",
            "half": "About 50",
            "right": "About 15",
        },
        reveal="**15 of the 100.** Only 1 discussed possible harm. What the papers did value most often were performance, generalization, quantitative evidence, efficiency, building on past work and novelty (Birhane and colleagues, 2022).",
    )
    guess_need
    return (guess_need,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Codes of conduct.** The ACM, a large international society of computing professionals, updated its *Code of Ethics and Professional Conduct* in 2018. Its first principles read: 1.1 contribute to society and to human well-being; 1.2 avoid harm; 1.3 be honest and trustworthy; 1.4 be fair and take action not to discriminate; 1.6 respect privacy. Principle 3.1 asks leaders to make sure that the public good is the central concern in all computing work, and the code calls the public good the paramount consideration. The IEEE, a large association of electrical and computer engineers, opens its code with the promise to hold paramount the safety, health and welfare of the public. A code is not a law: breaking it is no crime. But it gives a professional something to point to when saying no, and a standard that others can hold them to.

        **Research is never free of values.** The *value-free ideal* says that science should leave social and moral values out. In 1953 the philosopher Richard Rudner gave an argument against it that is still discussed. No evidence makes a claim certain, so a scientist must decide how much evidence is enough, and that depends on how bad a mistake would be: you want more certainty that a medicine holds no deadly amount of a poison than that a batch of belt buckles has no faults. The philosopher Heather Douglas built on this in 2009. In machine learning the same holds for which questions you study, which data you use, and what you call the best model. Since 2020, NeurIPS, one of the biggest conferences, has asked authors to think about the effects of their work: first with a *broader impact statement* in every paper, and from 2021 with a checklist. In Europe, *responsible research and innovation* asks the same of research: think ahead about its effects, involve the public, and change course when needed.

        The picture shows a choice like Rudner's. A check that flags hateful messages on a forum, made up by the course, can be set strict or loose. A strict setting lets fewer hateful messages through but flags more harmless ones. Which setting is best? The counts cannot say: it depends on how bad a miss is compared with a false alarm, and that is a question of whose harm counts more.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **code of conduct** | rules that a profession or organization agrees on for how its members should act, such as the ACM Code of Ethics for computing professionals |
        | **value-free ideal** | the ideal that science should be free of social and moral values; philosophers of science argue that values always play a part, for example in deciding how much evidence is enough |
        """),
        L.watch_out("Counting mistakes is not neutral either. Putting a cost on each kind of mistake makes the values visible, but someone still chose the costs: write down who, and why."),
        mo.Html(C.picture({"title": "What each setting gets wrong in 1,000 messages", "y": "messages",
                           "misses": "hateful messages let through", "alarms": "harmless messages flagged",
                           "alt": "Three settings of a check for hateful messages, tested on 1,000 messages. Strict lets 4 hateful messages through and flags 60 harmless ones. Middle lets 10 through and flags 25. Loose lets 25 through and flags 8."})),
        mo.md(r"""
        Counts made up by the course. The stricter the setting, the fewer hateful messages get through, and the more harmless messages are flagged by mistake.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To account for a choice in your work:

    1. Name the choice: what to measure, which data, which setting, what to publish.
    2. Name who gains and who pays when it goes wrong.
    3. Look up what your code of conduct says about it, such as 1.2 *avoid harm*.
    4. Make the values visible, for example as a cost for each kind of mistake.
    5. Write down what you chose and why, so that others can check it and disagree.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The check of the picture. `misses` holds how many hateful messages each setting lets through, `alarms` how many harmless messages it flags. The forum's moderators count a miss as 10 times as bad as a false alarm, because a hateful message hurts the people it is aimed at: `miss_cost` is 10 and `alarm_cost` is 1. The cost of a setting is the cost of its misses plus the cost of its false alarms. Run the cell.
    """)
    return


@app.cell
def example_moderators():
    _misses = {"strict": 4, "middle": 10, "loose": 25}
    _alarms = {"strict": 60, "middle": 25, "loose": 8}
    _miss_cost = 10
    _alarm_cost = 1

    for _setting in _misses:
        print(_setting, _miss_cost * _misses[_setting] + _alarm_cost * _alarms[_setting])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With these costs, `strict` is cheapest: 100, against 125 and 258. The people who write on the forum may see it differently: a harmless message that is taken down silences them. Exercise 1 counts with their costs.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cost():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Whose costs?

        The same check, but now with the costs of the writers: a false alarm counts 3 times as bad as a miss. `misses` and `alarms` hold the counts of the worked example, `miss_cost` and `alarm_cost` what one mistake of each kind costs, and the loop goes through the settings with `setting`. Replace the `...` with one line, as in the worked example, but put each setting's cost in `cost` instead of printing it. The last line is `cost`, so the check sees it. The check also tries other costs.
        """),
        L.task("Put in `cost`, under each setting, the cost of its misses plus the cost of its false alarms."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cost():
    box_cost = L.code_box("cost", lines=12)
    box_cost
    return (box_cost,)


@app.cell(hide_code=True)
def check_cost(box_cost):
    result_cost = L.check("cost", box_cost.value)
    result_cost
    return (result_cost,)


@app.cell(hide_code=True)
def hints_cost():
    L.hints(
        "cost",
        [
            "The sum of the worked example, with its answer stored under the setting.",
            "Each count is multiplied by what one such mistake costs. Which of the two costs belongs to the misses?",
            "`cost[setting] = ___ * misses[setting] + ___ * alarms[setting]`",
        ],
        why="`cost[setting] = miss_cost * misses[setting] + alarm_cost * alarms[setting]` gives `{'strict': 184, 'middle': 85, 'loose': 49}`: with the writers' costs, `loose` is cheapest.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_principles():
    mo.md(r"""
    ### Exercise 2. Which principle?

    Six situations an engineer might meet. Put each one under the principle of the ACM code it is mainly about.
    """)
    return


@app.cell(hide_code=True)
def sort_principles_cell():
    sort_principles = L.sort("principles")
    sort_principles
    return (sort_principles,)


@app.cell(hide_code=True)
def check_principles(sort_principles):
    result_principles = L.check("principles", sort_principles)
    result_principles
    return (result_principles,)


@app.cell(hide_code=True)
def hints_principles():
    L.hints(
        "principles",
        [
            "Ask for each situation: is someone put at risk, treated unequally, or is something about them kept or read?",
            "Two situations are about people who could get hurt. Two others are about groups that are treated worse than others.",
            "Avoid harm: the bridge and \\_\\_\\_. Be fair: \\_\\_\\_ and the loan. Respect privacy: \\_\\_\\_ and \\_\\_\\_.",
        ],
        why="**Avoid harm:** the bridge and the diet. **Be fair:** the accent and the loan. **Respect privacy:** the location and the messages.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_values():
    mo.md(r"""
    ### Exercise 3. Only numbers?
    """)
    return


@app.cell(hide_code=True)
def choice_values_cell():
    choice_values = L.choice(
        "values",
        question="A researcher writes: \"My paper only shows that my model is more accurate. There are no values in it.\" What does this lesson say?",
        options={
            "numbers": "That is right: accuracy is a number, so it is free of values",
            "ethics": "Values only play a part in papers about ethics or law",
            "right": "Choosing what to measure was already a value choice",
        },
        correct="right",
        why={
            "numbers": "The number may be measured well, but someone chose to measure accuracy, on these data, and to call the most accurate model the best. See *The idea*.",
            "ethics": "Rudner's argument holds for every claim that rests on evidence: how much evidence is enough depends on what a mistake would cost. See *The idea*.",
            "right": "What to measure, on which data, and what to call best are choices, and accuracy treats every mistake as equally bad, which is itself a value.",
        },
    )
    choice_values
    return (choice_values,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Principles and guidelines](/course/open/rai-principles?lang=en) you met principles that organizations write for AI. A code of conduct does the same for a profession: it says what you owe the public, whoever you work for. And the costs in Exercise 1 showed that even a technical choice, which setting to use, rests on values that you can name and account for. In the next lesson, *Consent and licenses of data*, you ask whether you may use a set of data at all.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_values, result_cost, result_principles):
    L.wrap_up(result_cost, result_principles, choice_values)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
