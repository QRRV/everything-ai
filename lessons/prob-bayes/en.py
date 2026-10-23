# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson prob-bayes. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow Bayes' theorem

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("prob-bayes", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Conditional probability](/course/open/prob-conditional?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rain():
    quick_rain = L.choice(
        "quick-rain",
        question="Of 40 rainy days, 30 began with a cloudy morning. What is P(cloudy morning | rain)?",
        options={
            "count": "30: the cloudy mornings",
            "rest": "0.25",
            "right": "0.75",
        },
        correct="right",
        why={
            "count": "A probability is a share between 0 and 1: count among the 40 rainy days only. See [Conditional probability](/course/open/prob-conditional?lang=en).",
            "rest": "That is the share of rainy days that began without clouds. See [Conditional probability](/course/open/prob-conditional?lang=en).",
            "right": "Among the 40 rainy days, 30 began cloudy: 30 / 40 = 0.75.",
        },
    )
    quick_rain
    return (quick_rain,)


@app.cell(hide_code=True)
def quick_check_turn():
    quick_turn = L.choice(
        "quick-turn",
        question="You know that P(A | B) = 0.6. What do you know about P(B | A)?",
        options={
            "right": "Nothing yet: it counts among other cases",
            "same": "It is 0.6 too: it is about the same A and B",
            "rest": "It is 0.4: together they make 1",
        },
        correct="right",
        why={
            "same": "P(A | B) counts among the cases with B, P(B | A) among the cases with A: those are other groups. See the watch-out in [Conditional probability](/course/open/prob-conditional?lang=en).",
            "rest": "0.4 is P(not A | B): the rest of the same group. See [Conditional probability](/course/open/prob-conditional?lang=en).",
            "right": "The condition decides which cases you count among, so one does not give the other. In this lesson you find out what else you need to turn it around.",
        },
    )
    quick_turn
    return (quick_turn,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Often you know a probability the wrong way round. The maker of a medical test says how often it is positive for people who are ill: P(positive | ill). But someone with a positive test wants to know P(ill | positive), and there is no table of counts to count it the other way, as in [Conditional probability](/course/open/prob-conditional?lang=en): only a few probabilities. Bayes' rule turns them around.

    **An everyday example.** A doctor knows how an illness shows: most people who have it get a fever. A patient comes in with a fever, and the doctor has to reason the other way round: how likely is the illness, now that there is a fever?
    """)
    return


@app.cell(hide_code=True)
def predict_test():
    guess_test = L.guess(
        "test",
        question="An illness is rare: 1 in 1000 people has it. A test for it is positive for 99% of the people who are ill, and wrongly positive for 1% of the healthy people. You test positive. How likely do you think it is that you are ill?",
        options={
            "high": "About 99%: the test is right 99 times in 100",
            "half": "About 50%",
            "right": "About 9%",
        },
        reveal=f"**About 9%.** Picture 100,000 people. {C.count(C.tested()[0], L.lang)} are ill, and {C.count(C.tested()[1], L.lang)} of them test positive. {C.count(C.tested()[2], L.lang)} are healthy, and 1% of them, {C.count(C.tested()[3], L.lang)}, test positive too. Of the {C.count(C.tested()[1] + C.tested()[3], L.lang)} positive tests, only {C.count(C.tested()[1], L.lang)} come from people who are ill: about 9%. The illness is so rare that the few false alarms among the many healthy people outnumber the right results among the few who are ill. It is the same effect as in exercise 3 of [Conditional probability](/course/open/prob-conditional?lang=en), now worked out from probabilities instead of counts.",
    )
    guess_test
    return (guess_test,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        How often A and B happen together can be worked out both ways round: as P(A | B) · P(B), the share of A among the cases with B, times how often B happens, or as P(B | A) · P(A). Both give P(A and B). Set them equal and divide by P(B): that is **Bayes' rule**, also called Bayes' theorem.
        """),
        L.formula(r"P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}"),
        mo.md(r"""
        P(A) is the **prior**: how likely A is before you know B. P(A | B) is the **posterior**: how likely A is after you know B. Bayes' rule updates the prior with the evidence B: it multiplies the prior by how likely the evidence is with A, and divides by how likely the evidence is at all.

        You often do not know P(B) directly. Add up the two ways B can happen, with A and without A; that is the **law of total probability**:
        """),
        L.formula(r"P(B) = P(B \mid A) \cdot P(A) + P(B \mid \text{not } A) \cdot P(\text{not } A)"),
        mo.md(r"""
        with P(not A) = 1 − P(A). For the test of the guess: P(positive) = 0.99 · 0.001 + 0.01 · 0.999 ≈ 0.011, and P(ill | positive) = 0.99 · 0.001 / 0.011 ≈ 0.09.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **Bayes' rule** | | P(A \| B) = P(B \| A) · P(A) / P(B): it turns a conditional probability around | `p_b_given_a * p_a / p_b` |
        | **prior** | | P(A): how likely A is before the evidence | `p_a` |
        | **posterior** | | P(A \| B): how likely A is after the evidence B | `bayes(...)` |
        | **base-rate fallacy** | | forgetting the prior: thinking a positive result for something rare is almost sure | |
        | **law of total probability** | | P(B) as the sum of the ways B can happen: with A, and without A | `p_b` |
        | P(not A) | P of not A | the probability that A does not happen: 1 − P(A) | `p_not_a` |
        """),
        L.watch_out("Forgetting the prior is a common mistake, called the **base-rate fallacy**: a test that is right 99 times in 100 makes a positive result sound 99% sure. But when the illness is rare, most positive results are false alarms. The prior, how common the illness is, belongs in the calculation: on top, and in P(B)."),
        mo.md(r"""
        The play block: 100,000 people and the test of the guess. The top bar is all of them, the bottom bar their positive tests, and the filled part is the people who are ill. Move the slider: it sets how many of the 100,000 are ill, and the test stays the same. The counts are rounded to whole people.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_prior_cell():
    _labels = {"title": "Who is ill, before and after a positive test?", "everyone": "All 100,000 people",
               "positive": "Their {pos} positive tests", "ill": "ill: {ill} ({share})", "healthy": "healthy: {healthy}",
               "true_pos": "from ill people: {true_pos} ({posterior})", "false_pos": "from healthy people: {false_pos} ({false_share})",
               "alt": "Two bars. Of all 100,000 people, {ill} are ill, {share}. Of their {pos} positive tests, {true_pos} come from people who are ill, {posterior}, and {false_pos} from healthy people."}

    def _frame(_ill):
        _said = "**{ill}** of the 100,000 people are ill, 1 in {one_in}: of the {pos} positive tests, {true_pos} come from people who are ill, <span class=\"ea-num\">{posterior}</span>."
        return C.bars(_ill, _labels, L.lang), _said.format(**C.frame(_ill, L.lang))

    _explore = L.explore("prior", "ill per 100,000", _frame, start=100, values=C.ILL, role="",
                         task="Find how many of the 100,000 people must be ill before half of the positive tests come from people who are ill.")
    _explore
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    At 1000 of the 100,000, 1 in 100, half of the positive tests come from people who are ill: 990 people who are ill test positive, and 990 healthy people too. Below that, the false alarms among the many healthy people outnumber the right results; above it, the right results win. The test never changed: the prior alone moved the posterior, from about 1% at 1 in 10,000 to 99% at 1 in 2.

    ### Recipe

    To turn P(B | A) into P(A | B):

    1. Write down the prior P(A), and how likely the evidence B is with A and without A: P(B | A) and P(B | not A).
    2. Work out P(not A) = 1 − P(A).
    3. Add up the two ways B can happen: P(B) = P(B | A) · P(A) + P(B | not A) · P(not A).
    4. Bayes' rule: P(A | B) = P(B | A) · P(A) / P(B).
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A spam filter: 20% of the mails are spam. The word "free" is in 60% of the spam mails, and in 5% of the other mails. A new mail contains "free". How likely is it to be spam? The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_bayes():
    # 1. the prior, and how likely "free" is with and without spam
    _p_spam = 0.2
    _p_free_given_spam = 0.6
    _p_free_given_not_spam = 0.05
    # 2. the mails that are not spam
    _p_not_spam = 1 - _p_spam
    print(_p_not_spam)
    # 3. both ways "free" can happen
    _p_free = _p_free_given_spam * _p_spam + _p_free_given_not_spam * _p_not_spam
    print(_p_free)
    # 4. Bayes' rule
    _p_spam_given_free = _p_free_given_spam * _p_spam / _p_free
    print(_p_spam_given_free)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Of all mails, 0.6 · 0.2 = 0.12 are spam with "free", and 0.05 · 0.8 = 0.04 are other mail with "free": 0.16 in all. Of those, 0.12 / 0.16 = 0.75 are spam. The word "free" raised the probability of spam from the prior 0.2 to the posterior 0.75.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_news():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Evidence that says nothing
    """)
    return


@app.cell(hide_code=True)
def choice_news_cell():
    choice_news = L.choice(
        "no-news",
        question="A test is positive for 30% of the people who are ill, and for 30% of the healthy people too. Someone tests positive. What does that do to the probability that they are ill?",
        options={
            "lower": "It goes down: the test is wrong most of the time",
            "right": "Nothing: the posterior is the prior",
            "up": "It goes up, as after any positive test",
        },
        correct="right",
        why={
            "lower": "Work it out with Bayes' rule: P(B | A) and P(B) are both 0.3 here. See *The idea*.",
            "up": "A positive test only helps when it is more likely for ill people than for healthy ones. See *The idea*.",
            "right": "P(positive) = 0.3 · P(ill) + 0.3 · P(healthy) = 0.3, so Bayes' rule gives 0.3 · P(ill) / 0.3 = P(ill). A test that is just as often positive for everyone tells you nothing.",
        },
    )
    choice_news
    return (choice_news,)


@app.cell(hide_code=True)
def text_exercise_bayes():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Bayes' rule as a function

        `bayes(p_a, p_b_given_a, p_b_given_not_a)` gives P(A | B): from the prior P(A), and how likely B is with A and without A. Step 2 of the recipe is there. Replace the two `...` with steps 3 and 4. The check calls `bayes(0.001, 0.99, 0.01)`, the test of the guess, and that should give about `0.0902`. It also tries other probabilities.
        """),
        L.task("Make `bayes` add up the two ways B can happen, and then turn the probability around."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bayes():
    box_bayes = L.code_box("bayes", lines=6)
    box_bayes
    return (box_bayes,)


@app.cell(hide_code=True)
def check_bayes(box_bayes):
    result_bayes = L.check("bayes", box_bayes.value)
    result_bayes
    return (result_bayes,)


@app.cell(hide_code=True)
def hints_bayes():
    L.hints(
        "bayes",
        [
            "Steps 3 and 4 of the recipe, with the names of the function.",
            "`p_b` is a sum of two products: B with A, and B without A, each times how often its case happens. Then the function gives back P(B | A) times the prior, divided by `p_b`.",
            "`p_b = p_b_given_a * p_a + p_b_given_not_a * ___`, and `return p_b_given_a * ___ / p_b`.",
        ],
        why="With the test of the guess: `p_b` = 0.99 · 0.001 + 0.01 · 0.999 = 0.01098, and 0.99 · 0.001 / 0.01098 ≈ 0.0902.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_second():
    mo.md(r"""
    ### Exercise 3. A second test
    """)
    return


@app.cell(hide_code=True)
def choice_second_cell():
    choice_second = L.choice(
        "second-test",
        question="After the positive test of the guess, the probability that you are ill went from 0.001 to about 0.09. You do the same test once more, independently of the first, and it is positive again. Which prior goes into Bayes' rule for the second test?",
        options={
            "start": "0.001 again: how rare the illness is",
            "right": "About 0.09: what the first test gave",
            "test": "0.99: how often the test finds the illness",
        },
        correct="right",
        why={
            "start": "0.001 was your belief before any test; after the first one, you know more. See what the prior is in *The idea*.",
            "test": "0.99 is P(positive | ill): it goes into Bayes' rule too, but it is not the prior. See *The idea*.",
            "right": "The posterior of the first test is your belief now, so it is the prior of the second. Bayes' rule then gives 0.99 · 0.09 / (0.99 · 0.09 + 0.01 · 0.91) ≈ 0.91: two positive tests make the illness likely.",
        },
    )
    choice_second
    return (choice_second,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Conditional probability](/course/open/prob-conditional?lang=en) showed that P(A | B) is not P(B | A); Bayes' rule turns one into the other, with the prior P(A) and the law of total probability for P(B). A belief that is updated with evidence becomes the prior for the next evidence. Classic spam filters work this way, word by word: with *Show extra lessons* on, you build one in *Spam or not: naive Bayes*, after *Pairs you never saw: smoothing*. In the next lesson, *Kinds of learning*, you tell the kinds of learning apart by where the right answers come from.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_news, choice_second, result_bayes):
    L.wrap_up(choice_news, result_bayes, choice_second)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
