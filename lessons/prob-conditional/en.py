# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson prob-conditional. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("prob-conditional", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Probability: counting how often](/course/open/prob-basics?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_spam():
    quick_spam = L.choice(
        "quick-spam",
        question="8 of 20 mails are spam. What is P(spam)?",
        options={
            "right": "0.4",
            "count": "8",
            "inverse": "2.5",
        },
        correct="right",
        why={
            "count": "That is how many there are. A probability is a share: of what? See the formula in [Probability: counting how often](/course/open/prob-basics?lang=en).",
            "inverse": "That is 20 / 8, the other way round. A probability lies between 0 and 1. See *The idea* in [Probability: counting how often](/course/open/prob-basics?lang=en).",
            "right": "8 of the 20 mails: 8 / 20 = 0.4.",
        },
    )
    quick_spam
    return (quick_spam,)


@app.cell(hide_code=True)
def quick_check_rest():
    quick_rest = L.choice(
        "quick-rest",
        question="Each day it either rains or stays dry, and P(rain) = 0.3. What is P(dry)?",
        options={
            "same": "0.3",
            "right": "0.7",
            "one": "1",
        },
        correct="right",
        why={
            "same": "The two outcomes need not be equally likely. What do all probabilities of a distribution add up to? See *The idea* in [Probability: counting how often](/course/open/prob-basics?lang=en).",
            "one": "1 is the probability of all outcomes together. What is left for dry? See *The idea* in [Probability: counting how often](/course/open/prob-basics?lang=en).",
            "right": "The two probabilities add up to 1: 1 − 0.3 = 0.7.",
        },
    )
    quick_rest
    return (quick_rest,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A spam filter does not care how likely spam is in general. It wants to know how likely this mail is spam, now that it has seen the word "free" in it. Knowing something changes the probability, and counting shows by how much.

    **An everyday example.** It rains on 40% of the days. But on a morning with dark clouds, you look only at the mornings like that one, and on a far bigger share of those days it rains.
    """)
    return


@app.cell(hide_code=True)
def predict_free():
    guess_free = L.guess(
        "free",
        question="Of 100 mails, 20 are spam. The word \"free\" is in 15 of the spam mails, and in 10 of the others. A new mail contains \"free\". What do you think is the probability that it is spam?",
        options={
            "overall": "0.2",
            "swapped": "0.75",
            "given": "0.6",
        },
        reveal="**0.6.** 25 mails contain \"free\", and 15 of those are spam: 15 / 25 = 0.6. 0.2 is P(spam) over all 100 mails, and 0.75 is the other way round: the share of the spam mails that contain \"free\".",
    )
    guess_free
    return (guess_free,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A conditional probability counts only among the cases where something else happened. Keep only the cases where B happened, and among them, take the share where A happened too.
        """),
        L.formula(r"P(A \mid B) = \frac{\text{how often } A \text{ and } B \text{ both happened}}{\text{how often } B \text{ happened}}"),
        mo.md(r"""
        The vertical bar is read as "given": everything after it is what you already know.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **conditional probability** | | the probability of $A$ among only the cases where $B$ happened | `both / with_b` |
        | $P(A \mid B)$ | P of A given B | the conditional probability of $A$, given $B$ | |
        """),
        L.watch_out("P(A | B) is not P(B | A). In the mails, P(spam | free) = 15 / 25 = 0.6, but P(free | spam) = 15 / 20 = 0.75: the condition decides which cases you count among."),
        mo.Html(C.picture({"title": "100 mails", "spam": "spam", "normal": "not spam", "total": "total", "free": "with \"free\"", "plain": "without", "caption": "P(spam | free) = 15 / 25 = 0.6",
                           "alt": "A table of 100 mails. Spam: 15 with the word free, 5 without, 20 in total. Not spam: 10 with free, 70 without, 80 in total. Together: 25 with free, 75 without, 100. The column with free is marked: P(spam given free) is 15 out of 25, 0.6."})),
        mo.md(r"""
        The condition "free" keeps one column: 25 mails. Among those, 15 are spam. The other 75 mails play no part.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find P(A | B) by counting:

    1. Keep only the cases where B happened.
    2. Count them: that is the number you divide by.
    3. Among them, count the cases where A happened too.
    4. Divide the count of step 3 by the count of step 2.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Ten days, each a pair: the sky in the morning and the weather later that day. `for _sky, _weather in _days:` gives the two values of each pair their own name, as `x, y = point` did in [Tuples, sets and sorting](/course/open/py-tuples-sets?lang=en). First P(rain) over all days, then P(rain | clouds). The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_rain():
    _days = [("clouds", "rain"), ("clouds", "rain"), ("clouds", "dry"), ("sun", "dry"), ("clouds", "rain"),
             ("sun", "dry"), ("sun", "dry"), ("clouds", "dry"), ("sun", "rain"), ("sun", "dry")]
    # P(rain), over all days
    _rain = 0
    for _sky, _weather in _days:
        if _weather == "rain":
            _rain = _rain + 1
    print(_rain)
    print(_rain / len(_days))
    # 1. and 2. keep and count the days with clouds
    _clouds = 0
    _both = 0
    for _sky, _weather in _days:
        if _sky == "clouds":
            _clouds = _clouds + 1
            # 3. among them, the days with rain
            if _weather == "rain":
                _both = _both + 1
    print(_clouds)
    print(_both)
    # 4. divide
    print(_both / _clouds)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    It rained on 4 of the 10 days: P(rain) = 0.4. Of the 5 mornings with clouds, 3 brought rain: P(rain | clouds) = 0.6. Knowing about the clouds raises the probability of rain from 0.4 to 0.6.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_studied():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Did studying help?
    """)
    return


@app.cell(hide_code=True)
def choice_studied_cell():
    choice_studied = L.choice(
        "studied",
        question="50 students: 30 studied, and 24 of those passed. Of the 20 who did not study, 8 passed. What is P(passed | studied)?",
        options={
            "joint": "0.48",
            "overall": "0.64",
            "right": "0.8",
        },
        correct="right",
        why={
            "joint": "That is 24 / 50: the share of all students who studied and passed. Which students does the condition keep? See the recipe.",
            "overall": "That is 32 / 50, P(passed) over all students. The condition is that they studied: see step 1 of the recipe.",
            "right": "Keep the 30 who studied; 24 of them passed: 24 / 30 = 0.8. Of the 20 who did not study, 8 passed: 8 / 20 = 0.4. Those who studied passed twice as often.",
        },
    )
    choice_studied
    return (choice_studied,)


@app.cell(hide_code=True)
def text_exercise_given():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. P(a | b) as a function

        Finish `p_given(cases, a, b)`. `cases` is a list of pairs: an outcome and a condition, such as `("spam", "free")`. The function gives back P(outcome is `a` | condition is `b`). Replace the `...`. The check calls `p_given(mails, "spam", "free")` with seven mails, which should give `0.75`. It also tries other cases.
        """),
        L.task("Make `p_given` give back the conditional probability."),
    ])
    return


@app.cell(hide_code=True)
def exercise_given():
    box_given = L.code_box("p-given", lines=10)
    box_given
    return (box_given,)


@app.cell(hide_code=True)
def check_given(box_given):
    result_given = L.check("p-given", box_given.value)
    result_given
    return (result_given,)


@app.cell(hide_code=True)
def hints_given():
    L.hints(
        "p-given",
        [
            "The recipe, with two counts in one loop, as in the worked example.",
            "Go through the pairs with `for outcome, condition in cases:`. Which pairs count for the number you divide by, and which for the number on top?",
            "`if condition == b:` adds 1 to `with_b`, and inside it `if outcome == ___:` adds 1 to `both`; at the end `return both / ___`.",
        ],
        why="The loop counts the cases whose condition is `b`, and among those the ones whose outcome is `a`. For the seven mails: 4 contain \"free\", 3 of them are spam, so 3 / 4 = 0.75.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_test():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A test, both ways

        A test for an illness, done on 1000 people. `counts` holds how many people fall in each group: ill or healthy, and a positive or negative test. Each key is a pair, such as `("ill", "positive")`, and `counts[("ill", "positive")]` gives the count of that group. Put P(ill | positive) in `p_ill_given_pos`, and P(positive | ill) in `p_pos_given_ill`. Replace both `...`. The last line shows both, so the check sees them. The check also tries other counts.
        """),
        L.task("Work out the conditional probability both ways."),
    ])
    return


@app.cell(hide_code=True)
def exercise_test():
    box_test = L.code_box("test-both-ways", lines=6)
    box_test
    return (box_test,)


@app.cell(hide_code=True)
def check_test(box_test):
    result_test = L.check("test-both-ways", box_test.value)
    result_test
    return (result_test,)


@app.cell(hide_code=True)
def hints_test():
    L.hints(
        "test-both-ways",
        [
            "For each one, step 1 of the recipe first: which two groups does its condition keep?",
            "P(ill | positive) keeps the positive tests, of ill and of healthy people. P(positive | ill) keeps the people who are ill, with a positive or a negative test. What goes on top, in both?",
            "`counts[(\"ill\", \"positive\")] / (counts[(\"ill\", \"positive\")] + counts[(___, ___)])`, and the same shape for the other one.",
        ],
        why="Both put the 9 people who are ill and test positive on top. P(ill | positive) divides by all 108 positive tests: about 0.083. P(positive | ill) divides by the 10 people who are ill: 0.9.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Probability: counting how often](/course/open/prob-basics?lang=en) you counted among all tries; a conditional probability counts among only the cases where the condition holds, and which way round matters. Later in the course, a model that tells spam from other mail gives exactly this: the probability of spam, given the words in the mail. In the next lesson, *Expected value and variance*, you work out what an outcome that depends on chance gives on average, and how far the results spread around that average.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_studied, result_given, result_test):
    L.wrap_up(choice_studied, result_given, result_test)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
