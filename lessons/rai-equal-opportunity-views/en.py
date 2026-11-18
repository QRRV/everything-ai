# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-equal-opportunity-views. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-equal-opportunity-views", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Why is discrimination wrong?](/course/open/rai-why-discrimination-is-wrong?lang=en)
    """)
    return


@app.cell(hide_code=True)
def quick_check_respect():
    quick_respect = L.choice(
        "quick-respect",
        question="Which answer says that discrimination is wrong when it does not look at what a person has made of their own life?",
        options={
            "right": "Eidelson's, about respect for persons",
            "harm": "Lippert-Rasmussen's, about the harm it does",
            "demean": "Hellman's, about demeaning",
        },
        correct="right",
        why={
            "right": "Eidelson asks you to respect people as equals who shape their own lives.",
            "harm": "Lippert-Rasmussen looks at the harm, more so to the worse off. See *The idea* in [Why is discrimination wrong?](/course/open/rai-why-discrimination-is-wrong?lang=en)",
            "demean": "Hellman looks at what an act expresses about a group. See *The idea* in [Why is discrimination wrong?](/course/open/rai-why-discrimination-is-wrong?lang=en)",
        },
    )
    quick_respect
    return (quick_respect,)


@app.cell(hide_code=True)
def quick_check_trust():
    quick_trust = L.choice(
        "quick-trust",
        question="In Phelps's model, the employer's trust in the test drops from 0.5 to 0.2. What happens to the gap between two people with the same score?",
        options={
            "right": "It grows, as the group counts more",
            "shrinks": "It shrinks",
            "same": "It stays the same, as the score is the same",
        },
        correct="right",
        why={
            "right": "The less the test counts, the more the group mean counts: 5 points became 8.",
            "shrinks": "When the test counts less, the group mean counts more, and pulls the two further apart. See Exercise 1 in [Why is discrimination wrong?](/course/open/rai-why-discrimination-is-wrong?lang=en)",
            "same": "The scores are the same, but the group means differ, and they count more when the test counts less. See Exercise 1 in [Why is discrimination wrong?](/course/open/rai-why-discrimination-is-wrong?lang=en)",
        },
    )
    quick_trust
    return (quick_trust,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A hiring model, a school admission or a loan decision is called fair when it gives everyone an equal chance. But what is an equal chance? The same test for everyone, or a test that takes into account where people started? The answer decides which fairness test you run on a model, and the tests of later lessons do not agree with each other.

    **An everyday example.** Two children take the same entrance exam. One had a tutor for a year; the other looked after younger brothers and sisters every afternoon. The exam is the same for both. Is the chance?
    """)
    return


@app.cell(hide_code=True)
def predict_warriors():
    guess_warriors = L.guess(
        "warriors",
        question="A society opens its class of warriors, once only for the rich, to everyone, and picks warriors by a test of strength. The poor are so badly fed that they lack the strength to win. Is the chance now equal?",
        options={
            "yes": "Yes, since the test is the same for everyone",
            "right": "Only on paper, as the poor start behind",
            "no": "No, a strength test is unfair",
        },
        reveal="**Only on paper.** This is the philosopher Bernard Williams's example (1962). The competition is open, and judged on what matters for the task, but the poor start too far behind to win it. The gap between an open competition and a real chance is what this lesson is about.",
    )
    guess_warriors
    return (guess_warriors,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Almost everyone agrees that people should get an equal chance. They disagree about what that means. Barocas, Hardt and Narayanan sort the answers into three views in their book on fairness.

        **The narrow view.** Treat people who are alike now in the same way. Judge each person only on what matters for the task, such as a test score, and give the place to whoever scores best. This is the old ideal of careers open to talents; philosophers call it formal equality of opportunity. If it leads to fewer hires from one group, the narrow view says the decision maker is not to blame. The book finds that it asks for little more than consistency.

        **The broad view.** Organize society so that people with the same talent and the same willingness to use it get the same prospects, whatever family they were born into. The philosopher John Rawls called this fair equality of opportunity (*A Theory of Justice*, 1971). Talent and willingness may still make a difference; the place where a person started may not. This view is less about one decision than about schools, healthcare and the other institutions of a society.

        **The middle view.** A decision maker should sometimes treat people who now look different in the same way, when the difference comes from something that should not count, such as past injustice or bad luck. Texas guarantees a place at its state universities to the top 10 percent of every high school class: the best student of a poor school does not have to beat the best of a rich school on raw scores.

        **When does the competition start?** The economist John Roemer (*Equality of Opportunity*, 1998) says that the views differ in when the competition starts. He splits what shapes an outcome into circumstances, which a person cannot help, such as the income of their parents, and effort, which they are responsible for. Only effort should count. Because circumstances shape effort too, Roemer compares people by their rank among those with the same circumstances: whoever tries hardest among the poor counts as much as whoever tries hardest among the rich.

        **Each view, a different test.** Heidari, Loi, Gummadi and Krause (2019) show what this does to a test. Alice and Bob each have 5 years of education, Anna 3 and Ben 7; Alice and Anna are women, Bob and Ben men. If the years themselves count as effort, Alice and Bob should get the same prospects. If effort is measured by rank within the own group, as Roemer does, Alice, first among the women, should get the same prospects as Ben, first among the men, and Bob may end up below Alice. Later lessons turn such ideas into tests that compare groups; the book warns that "moral notions never map exactly to technical criteria."

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **equality of opportunity** | the ideal that people get an equal chance at a place or a good; the views differ on what an equal chance is |
        | **circumstances** | in Roemer's words, what shapes an outcome that a person cannot help, such as where they were born |
        | **effort** | in Roemer's words, what shapes an outcome that a person is responsible for |
        """),
        L.watch_out("The broad view does not ask for equal outcomes. Under Rawls's view, talent and willingness may still lead to different results; only the place where a person started should not."),
        mo.Html(C.picture({"title": "Three views of an equal chance",
                           "rows": ["Looks at", "The competition starts", "Example"],
                           "views": [{"name": "Narrow", "texts": ["what people can do now",
                                                                  "at the decision",
                                                                  "one bar for every applicant"]},
                                     {"name": "Middle", "texts": ["people within their own group",
                                                                  "earlier: some differences do not count",
                                                                  "the best 10 percent of every school"]},
                                     {"name": "Broad", "texts": ["the prospects of people with the same talent and willingness",
                                                                 "at birth",
                                                                 "good schools for every child"]}],
                           "alt": "Three cards. Narrow: looks at what people can do now; the competition starts at the decision; example: one bar for every applicant. Middle: looks at people within their own group; the competition starts earlier, as some differences do not count; example: the best 10 percent of every school. Broad: looks at the prospects of people with the same talent and willingness; the competition starts at birth; example: good schools for every child."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find out which equal chance a rule or a model gives:

    1. Write down what it compares people on, such as a score or years of experience.
    2. Ask whether a difference in that can come from circumstances that people cannot help.
    3. If it cannot, or you judge that it should not matter: the narrow view, with one bar for everyone, may be enough.
    4. If it can: compare people within the group that shares their circumstances, as the middle view does, for example the best of each school.
    5. Ask what would have to change before the decision, in schools or at work, for the starting points to come closer: the broad view.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The narrow view, with scores made up by the course. `_scores` holds the test scores of six students at each of two schools, from high to low. There is one bar for everyone: whoever scores at least `_bar` gets in. A loop inside a loop goes through each school, and through each score of that school. Run the cell.
    """)
    return


@app.cell
def example_narrow():
    # made-up test scores of six students at each school, from high to low
    _scores = {"North": [91, 88, 85, 82, 76, 70], "South": [83, 77, 74, 69, 65, 61]}
    # the narrow view: one bar for everyone
    _bar = 80

    _admitted = {}
    for _school in _scores:
        _admitted[_school] = []
        for _score in _scores[_school]:
            if _score >= _bar:
                _admitted[_school].append(_score)
    print(_admitted)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Four from North, one from South. Under the narrow view nobody is to blame for that. The middle view, as in Texas, takes the best of each school instead: Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_admitted():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The best of each school

        The middle view: the best `top` students of each school get in, whatever their score. The lists run from high to low. Replace the `...` with a loop over the schools, as in the worked example, with the names without their underscore: `admitted` and `school`. For each school, `admitted[school]` gets the first `top` scores of that school. The last line is `admitted`, so the check sees it. The check also tries other numbers.
        """),
        L.task("Work out `admitted`: the best `top` scores of each school."),
    ])
    return


@app.cell(hide_code=True)
def exercise_admitted():
    box_admitted = L.code_box("admitted", lines=9)
    box_admitted
    return (box_admitted,)


@app.cell(hide_code=True)
def check_admitted(box_admitted):
    result_admitted = L.check("admitted", box_admitted.value)
    result_admitted
    return (result_admitted,)


@app.cell(hide_code=True)
def hints_admitted():
    L.hints(
        "admitted",
        [
            "Use the outer loop of the worked example, `for school in scores:`. You need no `if` and no inner loop: the lists already run from high to low.",
            "A slice gives the first items of a list: `scores[school][:2]` gives the first 2 scores of one school. Use `top` instead of 2.",
            "In the loop: `admitted[school] = ___`.",
        ],
        why="`admitted = {}`, then `for school in scores:` with `admitted[school] = scores[school][:top]` under it. That gives `{'North': [91, 88], 'South': [83, 77]}`: South's 77 gets in, North's 85 does not.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_views():
    mo.md(r"""
    ### Exercise 2. Which view?

    Six claims. Sort each one by the view it belongs to.
    """)
    return


@app.cell(hide_code=True)
def sort_views_cell():
    sort_views = L.sort("views")
    sort_views
    return (sort_views,)


@app.cell(hide_code=True)
def check_views(sort_views):
    result_views = L.check("views", sort_views)
    result_views
    return (result_views,)


@app.cell(hide_code=True)
def hints_views():
    L.hints(
        "views",
        [
            "Ask of each claim: does it look at people as they are now, does one decision take their starting point into account, or is it about how a whole society is set up?",
            "The top of the class is the Texas rule, and the family a person is born into comes from Rawls.",
            "Narrow: open to talents and \\_\\_\\_. Middle: the top of the class and \\_\\_\\_.",
        ],
        why="**Narrow:** open to talents, and same score, same decision. **Middle:** the top of the class, and past injustice. **Broad:** better schools, and born anywhere.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rank():
    mo.md(r"""
    ### Exercise 3. Rank in the own group
    """)
    return


@app.cell(hide_code=True)
def choice_rank_cell():
    choice_rank = L.choice(
        "rank",
        question="Four made-up applicants. In group A, Cleo has 4 years of education and Dana 6; in group B, Eli has 6 and Finn 8. If effort is measured by rank within the own group, as Roemer does, who is level with Dana?",
        options={
            "eli": "Eli, with the same 6 years",
            "right": "Finn, first of group B",
            "cleo": "Cleo, also in group A",
        },
        correct="right",
        why={
            "eli": "Same years is the comparison when the years themselves count as effort. By rank, Eli is second in group B and Dana first in group A. See *Each view, a different test* in *The idea*.",
            "right": "Dana is first in group A and Finn first in group B: by rank within the own group they are level, even with 6 against 8 years.",
            "cleo": "Within one group the ranks differ: Dana is first, Cleo second. Rank compares across groups. See *When does the competition start?* in *The idea*.",
        },
    )
    choice_rank
    return (choice_rank,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Why is discrimination wrong?](/course/open/rai-why-discrimination-is-wrong?lang=en) a group average pulled apart two people with the same score; the narrow view forbids exactly that, by looking only at what matters for the task. The middle view goes further: it sometimes treats people with different scores the same, because the difference comes from where they started, as your code did with the best of each school. Later lessons turn these views into tests that compare groups. One of those tests is called *equal opportunity*; despite its name, it is one technical rule, and it matches the ideas of this lesson only under assumptions. The next lesson, *Screened out by a filter*, looks at what an automatic filter does to people with a disability.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_rank, result_admitted, result_views):
    L.wrap_up(result_admitted, result_views, choice_rank)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
