# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-why-discrimination-is-wrong. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-why-discrimination-is-wrong", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Direct and indirect discrimination](/course/open/rai-discrimination-law?lang=en) and [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_milder():
    quick_milder = L.choice(
        "quick-milder",
        question="A neutral rule hits one group particularly hard. Its aim is legitimate and it is appropriate, but a milder means would work too. What is it?",
        options={
            "direct": "Direct discrimination",
            "right": "Prohibited indirect discrimination",
            "fine": "Allowed, since two of the three answers are yes",
        },
        correct="right",
        why={
            "direct": "The rule names no ground, so it is not direct. See *The idea* in [Direct and indirect discrimination](/course/open/rai-discrimination-law?lang=en).",
            "right": "With a milder means at hand, the rule is not necessary, and one no is enough.",
            "fine": "All three answers must be yes. See *When indirect is allowed* in [Direct and indirect discrimination](/course/open/rai-discrimination-law?lang=en).",
        },
    )
    quick_milder
    return (quick_milder,)


@app.cell(hide_code=True)
def quick_check_theory():
    quick_theory = L.choice(
        "quick-theory",
        question="Which theory judges a choice by what it brings about, added up over everyone?",
        options={
            "right": "Consequentialism, which adds up outcomes",
            "duty": "Deontology",
            "virtue": "Virtue ethics, which asks about good character",
        },
        correct="right",
        why={
            "right": "Consequentialism adds up the outcomes for everyone affected.",
            "duty": "Deontology holds that some duties and rights count whatever the outcome. See *The idea* in [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).",
            "virtue": "Virtue ethics asks what a person of good character would do. See *The idea* in [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).",
        },
    )
    quick_theory
    return (quick_theory,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The law says when discrimination is forbidden, but not why it is wrong. That question matters for models. A model that judges each person by the average of their group can be right on average and mean no harm to anyone, and still be what many philosophers call wrongful discrimination. To say what is wrong with it, and what a fix has to repair, you need the reasons.

    **An everyday example.** A shop owner follows every teenager around the shop, because teenagers steal more often, the owner says. Even if that were true on average, the teenager who never stole anything is judged by a group, not by anything they did.
    """)
    return


@app.cell(hide_code=True)
def predict_taste():
    guess_taste = L.guess(
        "taste",
        question="Economists call judging a person by the average of their group statistical discrimination. Does their basic model assume that the employer dislikes the group?",
        options={
            "yes": "Yes, that dislike is what makes it discrimination",
            "little": "A little: some dislike, and some guessing",
            "right": "No, the employer only guesses as well as it can",
        },
        reveal="**No.** In the models of Edmund Phelps (1972) and Kenneth Arrow (1973) the employer dislikes no group, and in Phelps's basic case even believes the averages correctly. That is why economists call it rational. This lesson asks whether it can still be wrong.",
    )
    guess_taste
    return (guess_taste,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Philosophers give several answers to the question of what makes discrimination wrong. Three are named often.

        **It harms.** Kasper Lippert-Rasmussen (*Born Free and Equal?*, 2014) argues that discrimination is wrong, when it is, because of the harm it does, and that harm to people who are already worse off weighs more. Then discrimination has no wrong of its own: it is one way of doing harm. An objection: a firm fires someone for a sexist reason; the person finds a better job and never learns why. Nobody seems harmed, yet many find the firing wrong.

        **It demeans.** Deborah Hellman (*When Is Discrimination Wrong?*, 2008) argues that treating people differently by a trait is wrong when it demeans: when it expresses that a person or a group is of lower status, done by someone with enough power for that to land. Hellman's example: an employer that requires its female staff to wear makeup says that a woman's body is there for others to look at. What counts is what the act means in its society, not what the person who does it thinks. An objection: a classmate without any power over you can also demean you.

        **It disrespects a person.** Benjamin Eidelson (*Discrimination and Disrespect*, 2015) argues that discrimination is wrong when it fails to show people the respect owed to persons: they are of equal worth, and they shape their own lives. Treating someone as an individual does not forbid all generalizing, which is "commonplace and inevitable", as Eidelson wrote in 2013; it asks you to look at what a person has made of their own life. A fire department that turns down every woman, instead of giving every applicant a fitness test, never asks whether the applicant has built up the strength.

        **Keeping groups down.** Another answer looks at groups. Differences between groups can come from injustice in the past, and a decision that leans on them makes that injustice worse. Barocas, Hardt and Narayanan list six answers in their book on fairness, this one among them under the name compounding injustice, and write that none of them is a complete theory.

        **Statistical discrimination.** The economists Phelps and Arrow described employers who cannot see a person's real ability and use the average of the person's group to fill the gap: no dislike, only guessing. In Phelps's model, as Aigner and Cain wrote it in 1977, the estimate of a person is pulled toward the average of the group, and more so the less the test can be trusted: the estimate is (1 − trust) × group mean + trust × score. Two people with the same score then get different estimates, only because of their group. Economists add that in the basic model each worker is paid what the information predicts, so they do not call it discrimination in the economic sense. Many philosophers still find it wrong, even when the average is correct: a group is a needlessly coarse guess when you could look at the person; the difference between the groups may come from past injustice; and it does not look at what the person made of their own life. A model that learns from data about groups can do the same, without anyone deciding it.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **statistical discrimination** | using what is true on average for a person's group as a stand-in for what you cannot see about that person; in Phelps's model the estimate mixes the group mean with the person's own score |
        | **trust** | in the formula, how much the test counts, from 0 (not at all) to 1 (fully); Aigner and Cain write it as γ |
        """),
        L.watch_out("Statistical discrimination is not the same as being wrong about a group. The averages can be correct, and nobody needs to dislike anyone; the question is whether judging a person by them is right."),
        mo.Html(C.picture({"title": "Same score, different estimates",
                           "group": "group", "mean": "group mean", "score": "score",
                           "xlabel": "estimate of the employer", "decimal": ".",
                           "alt": "Two rows on a scale from 40 to 80. Both people score 70. Group A has mean 60, and the estimate is 65.0; group B has mean 50, and the estimate is 60.0. Each estimate lies halfway between the group mean and the score, because trust is 0.5."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find out what is wrong with a rule or a model that treats groups differently:

    1. Ask who loses what, and whether those who lose were already worse off: harm.
    2. Ask what the rule says about a group, in its society, and whether those who use it have power: demeaning.
    3. Ask whether the rule looks at what each person did and made of their life, or only at their group: respect for the person.
    4. Ask where the difference between the groups comes from, and whether leaning on it makes an old injustice worse.
    5. Write down which answers apply: they tell you what a fix has to repair.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Phelps's model, with numbers made up by the course. Two people both score 70 on a test; `_means` holds the average score of their groups. `_trust` is 0.5: the test counts for half. The loop works out the estimate for a person of each group with `_group`. Run the cell.
    """)
    return


@app.cell
def example_estimates():
    # made-up numbers: one test score, the mean of each group, and how much the employer trusts the test
    _score = 70
    _means = {"A": 60, "B": 50}
    _trust = 0.5

    _estimates = {}
    for _group in _means:
        _estimates[_group] = (1 - _trust) * _means[_group] + _trust * _score
    print(_estimates)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The same score, but 65.0 against 60.0: 5 points apart, only because of the group. Now a test that the employer trusts less: Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_gap():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. How far apart?

        The same people, but the employer trusts the test less: `trust` is 0.2. Replace the `...` with the loop of the worked example, with the names without their underscore: `estimates` and `group`. Then work out `gap`, the estimate of `"A"` minus that of `"B"`. The last line is `gap`, so the check sees it. The check also tries other numbers.
        """),
        L.task("Work out `gap`: how many points apart two people with the same score end up."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gap():
    box_gap = L.code_box("gap", lines=11)
    box_gap
    return (box_gap,)


@app.cell(hide_code=True)
def check_gap(box_gap):
    result_gap = L.check("gap", box_gap.value)
    result_gap
    return (result_gap,)


@app.cell(hide_code=True)
def hints_gap():
    L.hints(
        "gap",
        [
            "Copy the loop of the worked example, without the `print`, and take the underscore off each name. One line comes after it.",
            "In the loop, each estimate is `(1 - trust)` times the group mean plus `trust` times the score. After the loop, subtract the two estimates.",
            "After the loop: `gap = estimates[\"A\"] - ___`.",
        ],
        why="`estimates = {}`, then `for group in means:` with `estimates[group] = (1 - trust) * means[group] + trust * score` under it, and after the loop `gap = estimates[\"A\"] - estimates[\"B\"]`. That gives `8.0`: 62 against 54.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_accounts():
    mo.md(r"""
    ### Exercise 2. Whose answer is it?

    Six claims. Sort each one by the answer it belongs to.
    """)
    return


@app.cell(hide_code=True)
def sort_accounts_cell():
    sort_accounts = L.sort("accounts")
    sort_accounts
    return (sort_accounts,)


@app.cell(hide_code=True)
def check_accounts(sort_accounts):
    result_accounts = L.check("accounts", sort_accounts)
    result_accounts
    return (result_accounts,)


@app.cell(hide_code=True)
def hints_accounts():
    L.hints(
        "accounts",
        [
            "Ask of each claim: is it about what someone loses, about what an act says about a group, or about looking at the person?",
            "The makeup rule is Hellman's own example, and the fire department comes from Eidelson.",
            "Harm: the cost and \\_\\_\\_. Demeaning: the makeup and \\_\\_\\_.",
        ],
        why="**It harms:** the cost and the worse off. **It demeans:** the makeup and the power. **It disrespects a person:** the fire department and the author of a life.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_average():
    mo.md(r"""
    ### Exercise 3. A correct average
    """)
    return


@app.cell(hide_code=True)
def choice_average_cell():
    choice_average = L.choice(
        "average",
        question="An employer knows, correctly, that people of group B score lower on average, and gives everyone from group B a lower estimate, whatever their own score. What do many philosophers in this lesson say?",
        options={
            "fine": "Nothing is wrong",
            "taste": "It is wrong only if the employer dislikes group B",
            "right": "It can be wrong, even with a correct average",
        },
        correct="right",
        why={
            "fine": "A correct average can still be a needlessly coarse guess, come from past injustice, and ignore what a person made of their own life. See *Statistical discrimination* in *The idea*.",
            "taste": "Dislike is not needed: statistical discrimination is about guessing, and the respect and harm answers still apply. See *The idea*.",
            "right": "The person is judged by the group, not by their own score, and the difference between the groups may come from past injustice.",
        },
    )
    choice_average
    return (choice_average,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en) you met consequentialism and deontology. The harm answer judges discrimination by its outcomes, with more weight for the worse off; the answers about demeaning and respect look at what an act says and how it treats a person, closer to deontology. In [Direct and indirect discrimination](/course/open/rai-discrimination-law?lang=en) the law looked at effects; here you saw why those effects can be wrong. The next lesson, *What is an equal chance?*, asks what it would mean to give everyone the same chance.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_average, result_accounts, result_gap):
    L.wrap_up(result_gap, result_accounts, choice_average)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
