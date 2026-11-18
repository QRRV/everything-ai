# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-discrimination-law. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-discrimination-law", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kind():
    quick_kind = L.choice(
        "quick-kind",
        question="A model gives women fewer loans than men with the same income. Which kind of harm is that?",
        options={
            "shown": "Representational harm, as it shows women as poorer",
            "right": "Allocative harm: women get fewer loans",
            "none": "No harm, if it is right on average",
        },
        correct="right",
        why={
            "shown": "Representational harm is about how a system shows or describes groups. A loan is something it hands out. See *The idea* in [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).",
            "right": "A loan is something the system hands out, and one group gets less of it than it should.",
            "none": "A model can be right on average and still hand out less to one group. See *The idea* in [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).",
        },
    )
    quick_kind
    return (quick_kind,)


@app.cell(hide_code=True)
def quick_check_measure():
    quick_measure = L.choice(
        "quick-measure",
        question="How do you measure allocative harm?",
        options={
            "accuracy": "By the accuracy of the model on all the people together",
            "right": "As the share of each group that gets a yes",
            "read": "By reading the outputs one by one and looking for stereotypes",
        },
        correct="right",
        why={
            "accuracy": "One number for everyone hides how each group fares. See *The idea* in [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).",
            "right": "You count, per group, how many people get a yes, and divide by the people in that group.",
            "read": "That is how you look for representational harm. See *The idea* in [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).",
        },
    )
    quick_measure
    return (quick_measure,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model hardly ever reads a column called *race* or *religion*. Yet it can treat groups very differently, through a postcode, a gap in a CV or a language test. Is that discrimination? Dutch law says it can be, even when nobody meant it, and whether a person or a computer made the choice does not matter.

    **An everyday example.** A club lets in only people taller than 1.80 meters. The rule names no sex, but far more men than women are that tall: without saying so, the club keeps out mostly women.
    """)
    return


@app.cell(hide_code=True)
def predict_vu():
    guess_vu = L.guess(
        "vu",
        question="In 2021 a student at the VU in Amsterdam complained that exam software often did not find the student's face, and linked that to skin color. What do you think the Netherlands Institute for Human Rights finally decided, in 2023?",
        options={
            "software": "The exam software discriminated against the student",
            "nothing": "The VU did nothing wrong",
            "right": "Not the software, but the complaint handling",
        },
        reveal="**Not the software, but the complaint handling.** In 2022 the Institute found enough facts to suspect indirect discrimination: the student's own problems, together with research showing that face detection in general works worse for people with a darker skin. Then the VU had to prove that it had not discriminated, and it could: the student had no more problems than other students, and those could be explained by a poor internet connection or glasses. But the complaint passed three bodies before one took it up, more than seven months later. That handling was prohibited distinction on grounds of race.",
    )
    guess_vu
    return (guess_vu,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Direct discrimination.** Someone is treated worse than another person in a comparable situation is, was or would be treated, because of a protected ground. An example from the Netherlands Institute for Human Rights: a hiring algorithm with variables that give women a lower recommendation.

        **Indirect discrimination.** A rule, criterion or practice that looks neutral puts people of a protected group at a particular disadvantage. The rule does not name the ground: you see it in the effect. Another example from the Institute: a job ad for a programmer that asks for perfect Dutch. That keeps out more people of non-Dutch origin, while a programmer does not need it; for a teacher it can be different.

        **The law.** In the Netherlands the Equal Treatment Act (AWGB) of 1994 forbids both, in work, in offering goods and services, and in education, among other places. It names eight grounds: religion, belief, political opinion, race, sex, nationality, heterosexual or homosexual orientation, and civil status. Race is read broadly, and also covers skin color, descent and national or ethnic origin. Disability and chronic illness have a law of their own, as does age at work; article 1 of the Constitution forbids discrimination on any ground. The EU directives of 2000 describe both kinds in the same words.

        **When indirect is allowed.** A neutral rule that hits one group harder is allowed only when it is objectively justified. That takes three questions, and every answer must be yes. Does the rule serve a *legitimate aim*? Is it an *appropriate* means to reach that aim? Is it *necessary*: is there no other means that makes less distinction, and does it stand in proportion to the aim? Direct discrimination has no such general excuse, only exceptions that the law lists one by one, for example work for which someone's sex is decisive.

        **The effect counts.** The Institute writes that in equal treatment law it does not matter whether a computer or a person makes the distinction: the effect is what counts. An employer cannot point to the maker of the software. And when someone brings facts that make discrimination plausible, the other side has to prove that it did not discriminate.

        **The Netherlands Institute for Human Rights.** Anyone who thinks they were discriminated against can ask the Institute (in Dutch: College voor de Rechten van de Mens) for a judgment, for free. Its judgments are not legally binding, there is no appeal, and it cannot impose a fine or damages: for that you go to court. But in more than 80 percent of the cases where it finds discrimination, measures are taken, by its own count. An organization can also ask about its own conduct. In 2023 the Institute judged a question from the dating app Breeze about its own self-learning algorithm, which Breeze suspected gave users with a darker skin color or a non-Dutch origin a lower chance of a match on average. The algorithm used no data on origin, but it made the way users liked each other stronger. The Institute called that indirect discrimination, and said that Breeze may, and even must, take measures.

        **A judgment on an automated rule.** In November 2024 the Institute judged the Dienst Toeslagen, which pays benefits. The population register had sent code 98, *no lawful residence*, for a woman's foster child, and the Dienst automatically stopped the woman's rent benefit and claimed it back. The code was wrong: the child had a valid residence permit. Code 98 names residence, not race, so this was no direct discrimination. But families without lawful residence almost always include someone of non-Dutch origin: indirect. The aim was legitimate and the means appropriate, but not necessary: stopping a benefit has acute and far-reaching consequences, so the Dienst should have checked the code first. That made it prohibited distinction on grounds of race.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **direct discrimination** | treating someone worse than others in a comparable situation because of a protected ground; Dutch law calls it *direct onderscheid* |
        | **indirect discrimination** | a rule or practice that looks neutral but puts people of a protected group at a particular disadvantage; in Dutch law *indirect onderscheid* |
        | **Equal Treatment Act** | the Dutch law (AWGB) that forbids both kinds on eight grounds, in work, services and education among other places |
        | **Netherlands Institute for Human Rights** | the Dutch body that judges complaints about unequal treatment, for free; its judgments are not binding, but they carry weight |
        """),
        L.watch_out("Indirect is not always forbidden: a neutral rule that hits one group harder is allowed when all three questions come out yes. And a rule is not fine just because it names no ground: indirect discrimination needs no intent, and no column called race."),
        mo.Html(C.picture({"title": "How the law looks at a rule",
                           "questions": ["Does the rule treat someone worse because of a protected ground?",
                                         "Does a neutral rule hit people of a protected group particularly hard?",
                                         "Indirect discrimination: is it objectively justified?"],
                           "ways": [["Yes: direct discrimination. Forbidden, except where the law lists an exception.",
                                     "No: go on to question 2."],
                                    ["Yes: indirect discrimination. Go on to question 3.",
                                     "No: no distinction."],
                                    ["Three times yes: allowed.", "One no: forbidden."]],
                           "steps": ["legitimate aim?", "appropriate?", "necessary?"],
                           "alt": "Three steps. 1: does the rule treat someone worse because of a protected ground? Yes means direct discrimination, forbidden except where the law lists an exception; no means go on. 2: does a neutral rule hit people of a protected group particularly hard? Yes means indirect discrimination; no means no distinction. 3: is the indirect discrimination objectively justified, with a legitimate aim, an appropriate means and a necessary means? Three times yes means allowed, one no means forbidden."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To test a rule, or a model, against equal treatment law:

    1. Check whether the rule itself treats people differently by a protected ground. If so, it is direct discrimination.
    2. Otherwise, count per group who the rule hits. If one group is hit particularly hard, it is indirect discrimination.
    3. Write down the aim of the rule, and ask whether that aim is legitimate.
    4. Ask whether the rule actually reaches that aim: is it appropriate?
    5. Ask whether another means makes less distinction, and whether the harm stands in proportion to the aim: is it necessary? One no, and the rule is forbidden.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first two questions of the picture, simplified, for a rule made up by the course. `_rule` says, with `"yes"` or `"no"`, whether the rule names a protected ground and whether it hits one group harder. `_kind` starts as `"no distinction"`; each `if` can change it, and the last one that comes true wins, so a rule that names a ground is always direct. Run the cell.
    """)
    return


@app.cell
def example_kind():
    # a made-up rule, described by the course
    _rule = {"names_ground": "no", "hits_harder": "yes"}

    _kind = "no distinction"
    if _rule["hits_harder"] == "yes":
        _kind = "indirect"
    if _rule["names_ground"] == "yes":
        _kind = "direct"
    print(_kind)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The rule names no ground but hits one group harder: indirect. Change `"names_ground"` to `"yes"` and run the cell again: direct, whatever the effect. For an indirect rule, the three questions under question 3 of the picture come next: Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_verdict():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Was the code-98 rule justified?

        The Institute found the code-98 rule of the Dienst Toeslagen indirect. `test` holds its answers to the three questions: `"aim"`, `"appropriate"` and `"necessary"`. Replace the `...` with lines in the form of the worked example: start `verdict` as `"allowed"`, and write an `if` for each question that makes `verdict` `"forbidden"` when the answer is `"no"`. The last line is `verdict`, so the check sees it. The check also tries other answers.
        """),
        L.task("Work out `verdict`: `\"allowed\"` or `\"forbidden\"`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_verdict():
    box_verdict = L.code_box("verdict", lines=11)
    box_verdict
    return (box_verdict,)


@app.cell(hide_code=True)
def check_verdict(box_verdict):
    result_verdict = L.check("verdict", box_verdict.value)
    result_verdict
    return (result_verdict,)


@app.cell(hide_code=True)
def hints_verdict():
    L.hints(
        "verdict",
        [
            "Indirect discrimination is allowed only when all three answers are yes. So start at allowed, and let every no change that.",
            "First `verdict = \"allowed\"`. Then three times an `if` that compares one answer in `test` with `\"no\"`, with `verdict = \"forbidden\"` under it.",
            "The first `if`: `if test[\"aim\"] == ___:` with `verdict = \"forbidden\"` under it; the other two the same, with `\"appropriate\"` and `\"necessary\"`.",
        ],
        why="`verdict = \"allowed\"`, then `if test[\"aim\"] == \"no\":` with `verdict = \"forbidden\"` under it, and the same for `\"appropriate\"` and `\"necessary\"`. That gives `'forbidden'`: the answer about `\"necessary\"` is no.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Exercise 2. Direct, indirect or neither?

    Six cases. Sort each one with the first two questions of the picture.
    """)
    return


@app.cell(hide_code=True)
def sort_kinds_cell():
    sort_kinds = L.sort("kinds")
    sort_kinds
    return (sort_kinds,)


@app.cell(hide_code=True)
def check_kinds(sort_kinds):
    result_kinds = L.check("kinds", sort_kinds)
    result_kinds
    return (result_kinds,)


@app.cell(hide_code=True)
def hints_kinds():
    L.hints(
        "kinds",
        [
            "For each case, ask first whether the rule itself names a protected ground, and only then whether it hits one group harder.",
            "Two cases name the ground itself: sex and skin color. Of the other four, one hits no group harder.",
            "Direct: the lower score and \\_\\_\\_. No distinction: \\_\\_\\_.",
        ],
        why="**Direct:** the lower score and the border check. **Indirect:** perfect Dutch, the match and students only. **No distinction:** the coding test.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_college():
    mo.md(r"""
    ### Exercise 3. What does a judgment do?
    """)
    return


@app.cell(hide_code=True)
def choice_college_cell():
    choice_college = L.choice(
        "college",
        question="The Institute finds that an employer's hiring tool discriminated. What does that judgment do by itself?",
        options={
            "fine": "The employer must pay a fine",
            "right": "It carries weight, but brings no fine or damages",
            "ban": "The tool may no longer be used anywhere in the country",
        },
        correct="right",
        why={
            "fine": "The Institute cannot impose a fine or damages; for that you go to court. See *The idea*.",
            "right": "Its judgments are not legally binding, and there is no appeal against them; for damages you go to court. But in more than 80 percent of the cases where it finds discrimination, measures are taken.",
            "ban": "The Institute judges one case and can make recommendations, but it cannot ban a tool. See *The idea*.",
        },
    )
    choice_college
    return (choice_college,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en) you measured allocative harm as the share of each group that gets a yes. That share is how you see indirect discrimination: a neutral rule whose yes comes much less often for one group. The next lesson, *The four-fifths rule*, turns those shares into one number to compare, on a real US table from 1994.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_college, result_kinds, result_verdict):
    L.wrap_up(result_verdict, result_kinds, choice_college)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
