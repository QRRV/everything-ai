# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-80-percent-rule. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-80-percent-rule", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Direct and indirect discrimination](/course/open/rai-discrimination-law?lang=en) and [One number per group: groupby](/course/open/data-groupby?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_indirect():
    quick_indirect = L.choice(
        "quick-indirect",
        question="A rule names no protected ground, but it hits women much harder than men. What is that called?",
        options={
            "direct": "Direct discrimination, as the effect is the same",
            "right": "Indirect discrimination",
            "none": "No discrimination, as long as nobody meant it",
        },
        correct="right",
        why={
            "direct": "Direct discrimination is when the rule itself treats people worse because of the ground. See *The idea* in [Direct and indirect discrimination](/course/open/rai-discrimination-law?lang=en).",
            "right": "A neutral rule that puts a group at a particular disadvantage is indirect discrimination; it is allowed only when it is objectively justified.",
            "none": "Indirect discrimination needs no intent: the effect counts. See *The idea* in [Direct and indirect discrimination](/course/open/rai-discrimination-law?lang=en).",
        },
    )
    quick_indirect
    return (quick_indirect,)


@app.cell(hide_code=True)
def quick_check_groupby():
    quick_groupby = L.choice(
        "quick-groupby",
        question="`df` is a table of people with a column `sex`. What does `df.groupby(\"sex\").size()` give?",
        options={
            "total": "The number of rows of the whole table, as one number",
            "right": "How many rows each value of `sex` has",
            "sorted": "The rows of the table, sorted by their value of `sex`",
        },
        correct="right",
        why={
            "total": "`groupby` first splits the rows into groups, so you get one number per group. See *The idea* in [One number per group: groupby](/course/open/data-groupby?lang=en).",
            "right": "One number per group: the rows with that value of `sex`.",
            "sorted": "`groupby` combines each group into one number. See *The idea* in [One number per group: groupby](/course/open/data-groupby?lang=en).",
        },
    )
    quick_groupby
    return (quick_groupby,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You can see indirect discrimination by counting who a rule says yes to. But how big must a difference between groups be before it is a signal? In the United States a rule of thumb has answered that since 1978, and much work on fairness in machine learning builds on it. In Europe there is no fixed number.

    **An everyday example.** 40 girls and 40 boys try out for a sports team; 12 girls and 24 boys make it. That is 30 percent against 60 percent: the girls get a yes half as often as the boys.
    """)
    return


@app.cell(hide_code=True)
def predict_women():
    guess_women = L.guess(
        "women",
        question="In a US table from 1994 (UCI Adult, also called Census Income), 30.4 percent of the men earn more than 50,000 dollars a year. What do you guess for the women?",
        options={
            "same": "About 28 percent, nearly as often as men",
            "two": "About 20 percent",
            "right": "About 11 percent, a third as often",
        },
        reveal="**About 11 percent**: 10.9. A model that learns to copy these labels gives women a yes about a third as often as men.",
    )
    guess_women
    return (guess_women,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The selection rate.** For each group, the number of people that get a yes divided by the number of people in that group: hired, a loan, a place on the team.

        **The four-fifths rule.** In 1978 four US federal agencies, among them the EEOC, published guidelines for selecting employees. One rule in them: a selection rate for any race, sex or ethnic group that is less than four-fifths, or 80 percent, of the rate of the group with the highest rate is generally regarded as evidence of adverse impact. Their questions and answers of 1979 give four steps: compute the rate of each group, find the highest, divide each rate by the highest, and check whether a ratio is below 0.8. Their example: of 80 White applicants 48 are hired, 60 percent, and of 40 Black applicants 12, 30 percent. 30 divided by 60 is 0.5: a sign of adverse impact.

        **Disparate impact.** The US idea behind the rule. In 1971, in *Griggs v. Duke Power*, the Supreme Court ruled that the law forbids practices that are "fair in form, but discriminatory in operation", unless they are related to the job. Duke Power required a high school diploma or passing a general intelligence test; in an EEOC case with the same tests, 58 percent of White applicants passed against 6 percent of Black applicants, a ratio of 0.10. In 1991 disparate impact was written into the law. Since April 2025 a presidential order tells US agencies to drop disparate impact as far as possible; the order does not change the law itself.

        **A rule of thumb.** The agencies write that the rule "is not intended as a legal definition", but is a practical means of keeping attention on serious differences. With small numbers it says little: 3 of 20 men and 1 of 10 women gives 0.67, too few to conclude adverse impact. And a smaller difference can still count when it is significant in statistical and practical terms.

        **In Europe.** EU law has no fixed number. In 1999 the Court of Justice asked whether a "considerably smaller percentage" of women than men could meet a rule; with 77.4 percent of men and 68.9 percent of women, it found that the figures did not seem, at first sight, to show that. The ratio is 0.89, so the four-fifths rule would not flag it either, but the Court used no ratio. In the Netherlands, the Institute for Human Rights found it enough for a suspicion of indirect discrimination that an employer let through all the men after an assessment and only one in three women.

        **The UCI Adult table.** This table holds 48,842 people from a US survey of 1994, with a label: earns more than 50,000 dollars a year or not. It has been used in more than three hundred papers about fairness. That line lies high: in 1994 it was above 89 percent of the women's incomes. A model that copies the labels gives women a yes about a third as often as men.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **selection rate** | the share of a group that gets a yes: the people selected from the group divided by the people in the group |
        | **four-fifths rule** | a rule of thumb: a group whose selection rate is below 80 percent of the highest rate is a signal; in Dutch the *80%-regel* |
        | **disparate impact** | in US law, a practice that looks neutral but affects a protected group much more than others; unlawful unless it is related to the job |
        | **adverse impact** | the guidelines' word for a clearly lower selection rate that works against a race, sex or ethnic group |
        | **impact ratio** | a group's selection rate divided by the highest selection rate; below 0.8 is the signal of the rule |
        """),
        L.watch_out("Passing the rule does not show that a process is fair, and failing it does not prove discrimination. It is a first signal that asks for a closer look: at the numbers behind it, and at whether the rule is justified."),
        mo.Html(C.picture({"title": "How often the labels of UCI Adult say yes",
                           "groups": {"Female": "women", "Male": "men"},
                           "ylabel": "selection rate", "line": "four-fifths of the highest",
                           "decimal": ".",
                           "alt": "Two bars: the share with a yes from the labels is 0.109 for women and 0.304 for men. A dashed line at four-fifths of the highest rate, 0.243, lies far above the bar of the women."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To apply the four-fifths rule to a model or a rule:

    1. Decide what counts as a yes: hired, a loan, earns more than 50,000 dollars.
    2. Per group, divide the people with a yes by all the people: the selection rate.
    3. Find the highest selection rate.
    4. Divide each rate by the highest: the impact ratio.
    5. Below 0.8 is a signal. Then look at how many people the rates rest on, and ask whether the rule is justified.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The labels of UCI Adult, by sex. `_picked` holds the rows with a yes: an income above 50,000 dollars. `_rates` divides the picked people per sex by all the people per sex: `groupby("sex").size()` counts the rows of each group, and the two counts are divided group by group. `_ratios` divides each rate by the highest, `_rates.max()`. Run the cell.
    """)
    return


@app.cell
def example_ratios():
    import pandas as pd

    # 48,842 people from a US survey of 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")

    _picked = _people[_people["income"] == ">50K"]
    _rates = _picked.groupby("sex").size() / _people.groupby("sex").size()
    _ratios = _rates / _rates.max()
    print(_rates.round(3))
    print(_ratios.round(3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The women get a yes 0.109 of the time, the men 0.304: a ratio of 0.36, far below 0.8. Now a rule that names no sex at all: Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ratios():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A rule about working hours

        A rule made up by the course: yes for whoever works at least `hours` hours a week, 45. Replace the `...` with the three lines of the worked example, with the names without their underscore: `picked`, `rates` and `ratios`. `picked` holds the rows where `hours_per_week` is at least `hours`. The last line is `ratios`, so the check sees it. The check also tries another number of hours.
        """),
        L.task("Work out `ratios`: the impact ratio of women and of men for the rule about hours."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ratios():
    box_ratios = L.code_box("ratios", lines=11)
    box_ratios
    return (box_ratios,)


@app.cell(hide_code=True)
def check_ratios(box_ratios):
    result_ratios = L.check("ratios", box_ratios.value)
    result_ratios
    return (result_ratios,)


@app.cell(hide_code=True)
def hints_ratios():
    L.hints(
        "ratios",
        [
            "The worked example picks the rows with a high income. Here you pick the rows with enough hours, and the other two lines stay the same.",
            "Pick with a comparison inside the brackets: the column `hours_per_week`, at least `hours`. Then the rates per sex, and divide them by the highest.",
            "The first line: `picked = people[people[\"hours_per_week\"] >= ___]`; the last: `ratios = rates / ___`.",
        ],
        why="`picked = people[people[\"hours_per_week\"] >= hours]`, then `rates = picked.groupby(\"sex\").size() / people.groupby(\"sex\").size()` and `ratios = rates / rates.max()`. That gives about `0.433` for women and `1.0` for men.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_flag():
    mo.md(r"""
    ### Exercise 2. Which groups does the rule flag?
    """)
    return


@app.cell(hide_code=True)
def choice_flag_cell():
    choice_flag = L.choice(
        "flag",
        question="In another example from the US agencies, the hiring rates are 60 percent for White applicants, 45 percent for American Indians, 48 percent for Hispanics and 51 percent for Black applicants. For which groups does the four-fifths rule signal adverse impact?",
        options={
            "right": "Only the group at 45 percent",
            "two": "The groups at 45 and 48 percent",
            "three": "All three groups",
        },
        correct="right",
        why={
            "right": "45 divided by 60 is 0.75, below 0.8. 48 divided by 60 is exactly 0.80, which is not less than four-fifths, and 51 divided by 60 is 0.85. The agencies reach the same answer.",
            "two": "48 divided by 60 is exactly 0.80. The rule signals a rate that is *less than* four-fifths of the highest. Divide each rate by 60.",
            "three": "Divide each rate by the highest, 60, and compare with 0.8: 51 divided by 60 is 0.85.",
        },
    )
    choice_flag
    return (choice_flag,)


@app.cell(hide_code=True)
def text_exercise_meaning():
    mo.md(r"""
    ### Exercise 3. What does passing tell you?
    """)
    return


@app.cell(hide_code=True)
def choice_meaning_cell():
    choice_meaning = L.choice(
        "meaning",
        question="A hiring tool gives every group at least 80 percent of the highest selection rate. What does that tell you?",
        options={
            "fair": "The tool is fair",
            "right": "Only that one rough signal is absent",
            "law": "It meets Dutch and EU law, which use the same rule",
        },
        correct="right",
        why={
            "fair": "The agencies write that the rule does not settle whether there is unlawful discrimination, and that smaller differences can still count. See *A rule of thumb* in *The idea*.",
            "right": "The rule only keeps attention on serious differences. A smaller difference can still matter, and with small numbers the rule says little either way.",
            "law": "EU and Dutch law have no fixed number: the Court of Justice asks whether a considerably smaller percentage of a group can meet a rule. See *In Europe* in *The idea*.",
        },
    )
    choice_meaning
    return (choice_meaning,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Direct and indirect discrimination](/course/open/rai-discrimination-law?lang=en) you saw that a neutral rule is indirect discrimination when it hits a group particularly hard; the impact ratio is one way to put a number on *particularly*. The counting is the `groupby` of [One number per group: groupby](/course/open/data-groupby?lang=en): one count per group, divided group by group. The next lesson, *Why is discrimination wrong?*, asks what makes such a difference wrong at all.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_flag, choice_meaning, result_ratios):
    L.wrap_up(result_ratios, choice_flag, choice_meaning)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
