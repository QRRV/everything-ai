# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-abstraction-traps. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-abstraction-traps", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Fairness in an organization](/course/open/rai-fairness-in-organizations?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_behind():
    quick_behind = L.choice(
        "quick-behind",
        question="Three groups have word error rates of 0.30, 0.20 and 0.26. Before looking, a team agreed on a limit of 0.05 above the best group. Which groups are behind?",
        options={
            "only": "0.30 only",
            "right": "0.30 and 0.26",
            "all": "All three, 0.20 too",
        },
        correct="right",
        why={
            "only": "0.26 lies above the line too: the best group is at 0.20, so the line lies at 0.20 + 0.05 = 0.25. See [Fairness in an organization](/course/open/rai-fairness-in-organizations?lang=en).",
            "right": "The best group is at 0.20, so the line lies at 0.25: 0.30 and 0.26 are above it.",
            "all": "0.20 is the best group itself, and it cannot lie above its own rate plus a limit. The line lies at 0.25. See [Fairness in an organization](/course/open/rai-fairness-in-organizations?lang=en).",
        },
    )
    quick_behind
    return (quick_behind,)


@app.cell(hide_code=True)
def quick_check_reactive():
    quick_reactive = L.choice(
        "quick-reactive",
        question="In the terms of Rakova and colleagues, which of these is reactive?",
        options={
            "right": "A fix after a news story",
            "review": "A review at set moments before launch",
            "board": "A board that can stop a system in time",
        },
        correct="right",
        why={
            "right": "Acting after bad press is reactive: the harm has already reached people. Set reviews and a board that can stop a system are proactive.",
            "review": "Set moments to review a system in advance are what makes work proactive. See [Fairness in an organization](/course/open/rai-fairness-in-organizations?lang=en).",
            "board": "Some proactive organizations have internal review boards; one that can stop a system acts before harm, not after it. See [Fairness in an organization](/course/open/rai-fairness-in-organizations?lang=en).",
        },
    )
    quick_reactive
    return (quick_reactive,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In 2011, Kentucky passed a law, House Bill 463, that made a pretrial risk assessment mandatory: a tool that scores each defendant's risk before trial. For defendants ranked low or moderate risk, the law set a default of release at once, without money. Ninety percent of defendants were ranked low or moderate risk. Megan Stevenson (2018) studied more than a million criminal cases. If judges had followed the default, 90 percent would have been released at once without money; in practice 29 percent were, at the first bail setting. Judges ignored the default in more than two-thirds of cases, and over time went back to how they set bail before. The law led to "only a trivial increase in pretrial release". And the gap between White and Black defendants in release without money jumped from about 2 to 10 percentage points. Not, Stevenson writes, "because the risk assessment was more racially biased than judicial discretion": judges in different regions responded differently to the law. A fairness check on the tool's scores would not have seen this gap.

    **An everyday example.** A school gives every pupil the same laptop. Measured by the laptops, that is as equal as it gets. But a pupil without internet at home can do less homework with it than a pupil with internet. Whether the plan works out fairly depends on the homes as well, not only on the laptops.
    """)
    return


@app.cell(hide_code=True)
def predict_release():
    guess_release = L.guess(
        "release",
        question="Kentucky's tool ranked 90 percent of defendants as low or moderate risk, and for them the law set a default of release without money. What share was released without money at the first bail setting?",
        options={
            "most": "Close to 90 percent",
            "two": "About two in three",
            "right": "Under a third",
        },
        reveal="**Under a third: 29 percent.** Had judges followed the default, 90 percent would have been released at once without money. Judges ignored the default in more than two-thirds of cases, and the change that came right after the law faded as they went back to how they set bail before.",
    )
    guess_release
    return (guess_release,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The system, not the tool.** Andrew Selbst, danah boyd, Sorelle Friedler, Suresh Venkatasubramanian and Janet Vertesi (2019) argue that "abstraction and modular design", bedrock ideas of computer science, can make technical fairness work "ineffective, inaccurate, and sometimes dangerously misguided" once it enters the society around a decision system. In their words: "Fairness and justice are properties of social and legal systems like employment and criminal justice, not properties of the technical tools within." The field of science and technology studies calls a system of technical and social parts together a *sociotechnical system*.

        **Three frames.** Where you draw the boundary decides what you can see. In the *algorithmic frame* there are only the choices of how to represent the data and what to label; there, "any notion of 'fairness' cannot even be defined." The *data frame* adds the algorithm's inputs and outputs: predictions and outcomes per group, where the metrics of this module look. The *sociotechnical frame* puts the people and institutions that act on the outputs inside the boundary, so that fairness becomes a property of the whole, from end to end. A pretrial tool's output is a risk; the court's output is that a defendant is released, released on bail, or detained.

        **Five traps.** Selbst and colleagues name five traps: ways to fail because the technical and the social parts are not seen together.

        - The *framing trap*: failing to model the entire system over which fairness should hold. Judges do not consistently take a tool's recommendation into account: some might lean on it too much, out of automation bias, others might deviate from it in biased ways or ignore it. A guarantee about the tool alone may then be incorrect.
        - The *portability trap*: moving a solution made for one social context to another. The culture of computer science "prizes and often demands portability", so the same classifier seems to fit recidivism, loan default and hiring. Even between court jurisdictions, local fairness concerns can differ enough that systems do not transfer well.
        - The *formalism trap*: missing that fairness is "procedural, contextual, and contestable", which no formula settles. For a résumé screen a false negative may matter less than a false positive, because the interview filters later; in criminal justice, false positives keep people locked up. And the four-fifths rule is only the first step of disparate impact in US employment law, which then asks whether a test is job related and consistent with business necessity.
        - The *ripple effect trap*: missing how a new tool changes the people and the values of the system it enters. Judges might defer to a score because it looks more rigorous, or resist it. A tool can quietly put what is measurable first: keeping someone locked up so that they cannot commit crimes is "only one of at least seven existing rationales for punishment".
        - The *solutionism trap*: failing to see that the best solution to a problem may not involve technology. When you start from the technology, "there is never an opportunity to evaluate whether the technology should be built in the first place."

        **Not a reason to stop measuring.** The way out, Selbst and colleagues write, is to move "from a solutions-oriented approach to a process-oriented one", with a boundary that includes "social actors, institutions, and interactions". They suggest going through the traps "perhaps in reverse order": first whether a technical solution fits the situation at all, last whether the frame includes the people and data that matter. For pretrial tools that means weighing the tool against policies that make it unnecessary, such as not detaining people charged with nonviolent crimes by default, and running a pilot to learn whether judges will follow it. "This does not mean that fair-ML is impossible to implement. It does, however, mean that not all problems can or should be solved with technology." Ben Green (2020) reports the same from practice: risk assessments "shift discretion to different people and places rather than eliminate discretion altogether"; judges in Cook County, Illinois, diverged from the pretrial risk assessment 85 percent of the time and released far fewer defendants than recommended.

        **Solutionism.** Evgeny Morozov (2013) borrowed the word from architecture and urban planning. For Morozov, solutionism recasts complex social situations as "neatly defined problems with definite, computable solutions", and what is contested is often not the solution but "their very definition of the problem itself." Selbst and colleagues and Ben Green write "technological solutionism"; *techno-solutionism* is a common short form. The trap of Selbst and colleagues is milder than the belief: not that people think a technical fix always works, but that they do not see that the best solution may not be technical.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **sociotechnical system** | a system seen as technology plus the people, organizations and rules around it |
        | **abstraction trap** | a way a fairness method fails because it leaves out part of the system it works in; Selbst and colleagues name five |
        | **framing**, **portability**, **formalism**, **ripple effect** and **solutionism trap** | the five: not modeling the whole system; moving a solution to another context; putting fairness in one formula; missing how a tool changes the people around it; not seeing that the best solution may not be technical |
        | **techno-solutionism** | short for technological solutionism: the belief that social problems are best solved with technology, even when their causes are political or social |
        """),
        L.watch_out("The traps are no argument against measuring. They say what else to measure: the decisions people make with a tool, not only its scores. And they say what to ask first: whether a tool is the answer at all."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Before you build or judge a fairness fix, go through the traps; Selbst and colleagues suggest, perhaps, this order, the reverse of their own:

    1. Solutionism: would a change that is not technical, such as a new rule or more staff, do better? If so, stop here.
    2. Ripple effect: how will the people and the organization behave once the tool is there?
    3. Formalism: which kind of fairness fits this context, and who gets a say in it?
    4. Portability: was the tool, or its threshold, made for this place and these people?
    5. Framing: draw the boundary around the tool and the people who act on its output, and measure their decisions, not only the tool's scores.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Ten cases in one court, made up by the course: what the tool recommends, and what the judge decided. `"release"` is release without money; `"bond"` is a money bond. The loop counts how often each of them says release. Run the cell.
    """)
    return


@app.cell
def example_court():
    # the tool's recommendation and the judge's decision in ten cases
    _cases = [
        {"tool": "release", "judge": "release"},
        {"tool": "release", "judge": "bond"},
        {"tool": "release", "judge": "bond"},
        {"tool": "release", "judge": "release"},
        {"tool": "bond", "judge": "bond"},
        {"tool": "release", "judge": "bond"},
        {"tool": "release", "judge": "bond"},
        {"tool": "release", "judge": "release"},
        {"tool": "release", "judge": "bond"},
        {"tool": "release", "judge": "bond"},
    ]

    _tool = 0
    _judge = 0
    for _case in _cases:
        if _case["tool"] == "release":
            _tool = _tool + 1
        if _case["judge"] == "release":
            _judge = _judge + 1
    print("tool", _tool / len(_cases))
    print("judges", _judge / len(_cases))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The tool would release 9 of the 10; the judges released 3. Judged by its scores, this tool releases nearly everyone; what happens to the defendants is decided by the judges. Does that happen alike for two groups? Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The tool and the judges

        `cases` holds 16 cases of one court, made up by the course after the pattern Stevenson found in Kentucky: each with the defendant's group, A or B, the county, the tool's recommendation and the judge's decision. Work out `share`: the share of the cases of `group` in which `key` says `"release"`. Count the cases of the group in `total`, and those of them with `case[key] == "release"` in `released`; then `share = released / total`. With `key = "judge"`, `case[key]` is `case["judge"]`. The check also tries the tool, and group A.
        """),
        L.task("Work out `share`: the share of the cases of `group` in which `key` says release."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("share", lines=33)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("share", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "share",
        [
            "Two counts in one loop over the cases: the cases of the group, and those of them that `key` says to release.",
            "Under `for case in cases:`, `if case[\"group\"] == group:` adds 1 to `total`; inside that, `if case[key] == \"release\":` adds 1 to `released`.",
            "After the loop: `share = ___ / total`.",
        ],
        why="`total = 0` and `released = 0`; then `for case in cases:` with `if case[\"group\"] == group:` and `total = total + 1`, and under it `if case[key] == \"release\":` with `released = released + 1`. After the loop, `share = released / total`. That gives `0.375`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_trap():
    mo.md(r"""
    ### Exercise 2. Which trap?
    """)
    return


@app.cell(hide_code=True)
def choice_trap_cell():
    choice_trap = L.choice(
        "trap",
        question="A company built a tool that screens job applications, and it meets a fairness criterion there. A court buys it, trains it on its own records, and uses it to predict whether defendants will appear for trial. Which trap is this?",
        options={
            "formalism": "The formalism trap",
            "right": "The portability trap",
            "ripple": "The ripple effect trap",
        },
        correct="right",
        why={
            "formalism": "The formalism trap is about putting fairness into one formula. Here a tool made for hiring is moved to a court. See *Five traps* in *The idea*.",
            "right": "A solution made for one social setting, hiring, is moved to another, a court. Even between courts, Selbst and colleagues write, local fairness concerns can differ enough that systems do not transfer well. Training it on the court's own records changes the data, not the social setting around the decision.",
            "ripple": "The ripple effect trap is about how a tool changes the people and values around it once it is in use. Here the problem starts earlier: a tool made for hiring is moved to a court. See *Five traps* in *The idea*.",
        },
    )
    choice_trap
    return (choice_trap,)


@app.cell(hide_code=True)
def text_exercise_overrides():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Where does the gap come from?

        The tool recommends releasing both groups alike, the judges do not. Work out `overrides`: a dict from each county in `counties` to the share of its cases in which the judge decided otherwise than the tool, `case["judge"] != case["tool"]`. Start with `overrides = {}`. For each county, set `total = 0` and `changed = 0`, loop over the cases, and count the cases of the county in `total` and those of them where the judge decided otherwise in `changed`. Then `overrides[county] = changed / total`. The check also tries another court.
        """),
        L.task("Work out `overrides`: per county the share of its cases where the judge decided otherwise than the tool."),
    ])
    return


@app.cell(hide_code=True)
def exercise_overrides():
    box_overrides = L.code_box("overrides", lines=32)
    box_overrides
    return (box_overrides,)


@app.cell(hide_code=True)
def check_overrides(box_overrides):
    result_overrides = L.check("overrides", box_overrides.value)
    result_overrides
    return (result_overrides,)


@app.cell(hide_code=True)
def hints_overrides():
    L.hints(
        "overrides",
        [
            "As in exercise 1, with one more loop around it: for each county, two counts over the cases.",
            "Under `for county in counties:`, `total = 0` and `changed = 0`; then `for case in cases:` with `if case[\"county\"] == county:` adding 1 to `total`, and inside that `if case[\"judge\"] != case[\"tool\"]:` adding 1 to `changed`.",
            "After the inner loop, at the level of `for case`: `overrides[___] = changed / total`.",
        ],
        why="`overrides = {}`, then `for county in counties:` with `total = 0` and `changed = 0`, and `for case in cases:` with `if case[\"county\"] == county:`, `total = total + 1`, and under it `if case[\"judge\"] != case[\"tool\"]:` with `changed = changed + 1`. After the inner loop, `overrides[county] = changed / total`. That gives `{'rural': 0.0, 'urban': 0.5}`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_first():
    mo.md(r"""
    ### Exercise 4. What comes first?
    """)
    return


@app.cell(hide_code=True)
def choice_first_cell():
    choice_first = L.choice(
        "first",
        question="Selbst and colleagues suggest going through the five traps, perhaps in reverse order. Which question does a team then ask first about a new risk tool for a court?",
        options={
            "rates": "Are its error rates equal for every group?",
            "right": "Is a tool the right answer here at all?",
            "elsewhere": "Did it work where it was made?",
        },
        correct="right",
        why={
            "rates": "Error rates per group come later, in the formalism and framing traps. In the reverse order the first question is whether a tool fits the situation at all. See *Not a reason to stop measuring* in *The idea*.",
            "right": "The reverse order starts with the solutionism trap: whether a technical solution fits the situation at all. For pretrial tools, that means weighing the tool against policies that make it unnecessary.",
            "elsewhere": "That is the portability trap, fourth in the reverse order. The first question is whether a tool fits the situation at all. See *Not a reason to stop measuring* in *The idea*.",
        },
    )
    choice_first
    return (choice_first,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Fairness in an organization](/course/open/rai-fairness-in-organizations?lang=en) the question was when a team looks at fairness. Here it is what the team looks at. The metrics of this module measure a tool in the data frame; the gap in exercise 1 only showed when the judges' decisions were counted too, and exercise 3 found it in the counties, not in the tool. A fairness review should draw its boundary around the decisions people make with a tool, and ask first whether a tool is the answer at all.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_first, choice_trap, result_overrides, result_share):
    L.wrap_up(result_share, choice_trap, result_overrides, choice_first)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
