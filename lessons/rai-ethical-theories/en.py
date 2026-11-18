# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-ethical-theories. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-ethical-theories", "en")
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
def quick_check_measure():
    quick_measure = L.choice(
        "quick-measure",
        question="A model decides which applicants are invited for a job interview. How do you measure allocative harm?",
        options={
            "outputs": "Ask the model to write many job ads and count the stereotypes in them",
            "right": "Per group, the share of applicants that gets an invitation",
            "overall": "The share of all applicants that is invited",
        },
        correct="right",
        why={
            "outputs": "That is how you look for representational harm. An invitation is handed out: what can you count per group? See *The idea* in [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).",
            "right": "An invitation is handed out, so you count per group who gets it: the yes rate of each group.",
            "overall": "One number over everyone hides how the groups differ. What do you compare? See the formula in [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).",
        },
    )
    quick_measure
    return (quick_measure,)


@app.cell(hide_code=True)
def quick_check_look():
    quick_look = L.choice(
        "quick-look",
        question="Why do you have to look for stereotypes in what a system shows, instead of measuring them from its decisions?",
        options={
            "rare": "Because they are much rarer than allocative harm, so a test set seldom holds any",
            "pictures": "Because they only happen in pictures",
            "right": "Because a stereotype decides nothing, so there is no yes to count",
        },
        correct="right",
        why={
            "rare": "It is not about how often it happens. Is anyone turned down when a system shows a stereotype? See the *Watch out* box in [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).",
            "pictures": "A translation and a search box can show a stereotype too, in words. Is anyone turned down when they do? See *The idea* in [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).",
            "right": "Nobody is turned down by a stereotype, so you collect many outputs and count who is shown how.",
        },
    )
    quick_look
    return (quick_look,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Two people can agree on every number and still disagree about what a system should do. That is not because one of them cannot count: they judge by different things. Ethics has thought about those things for a long time, and its main theories each ask their own question. Knowing the questions helps you see what is really at stake in a choice, and say it in words others can follow.

    **An everyday example.** One slice of cake is left. Give it to whoever is hungriest, to whoever was promised it, or cut it so that everyone gets a piece: each answer follows a different idea of what is right.
    """)
    return


@app.cell(hide_code=True)
def predict_far():
    guess_far = L.guess(
        "far",
        question="A hospital has a program that keeps patients from coming back after they go home. A model picks the patients: either those who live nearby first, because they come to every session, or those who need it most first. Nearby first prevents 32 readmissions, need first 28. Compared with the patients near the hospital, how big a share of the patients far away do you think nearby first helps?",
        options={
            "same": "About the same share as near the hospital",
            "less": "A slightly smaller share than near the hospital",
            "right": "A fifth as large a share",
        },
        reveal="**A fifth as large a share.** Nearby first helps 2 of the 100 patients far away, 0.02, against 30 of the 300 nearby, 0.1. The policy that does the most good in total does the least for the group that is worst off. The numbers of this hospital are made up by the course.",
    )
    guess_far
    return (guess_far,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **AI ethics** asks what is right and wrong in how AI is made and used. It borrows its tools from ethics, which has several theories. Each asks a different question about the hospital's choice.

        - **Consequentialism**: what does each option bring about, added up over everyone? The best choice does the most good. Nearby first prevents 32 readmissions, need first 28.
        - **Deontology**: which duties and rights hold, whatever the outcome? Immanuel Kant held that you may never treat a person merely as a means. A deontologist can say that leaving out patients because they bring the total down treats them as numbers, not as people.
        - *A fair agreement* (contractualism): which rule could everyone accept? John Rawls asked you to choose behind a *veil of ignorance*, without knowing who you will be, and argued that you would then look after the worst off. Need first helps 0.06 of the worst-off group, nearby first 0.02.
        - **Virtue ethics**: what would a person of good character do? It looks for practical wisdom more than for one rule: a just and caring doctor.
        - **Care ethics**: who depends on whom? It starts from relationships and from the people who need others, and listens to what they say they need.

        Ethics did not grow in the West alone. In *Ubuntu*, from southern Africa, a person is a person through other people: it asks what a choice does to a community and the ties within it. Confucian ethics looks at roles and relationships, and at harmony between them. The theories often agree; here most of them point away from leaving out the patients far away. Where they do not agree, as with the most good in total against the most for the worst off, you see what is at stake.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **AI ethics** | the study of what is right and wrong in how AI is made and used, and what to do about it |
        | **consequentialism** | judging a choice by what it brings about: the best choice does the most good, added up over everyone |
        | **deontology** | judging a choice by duties and rights that hold whatever the outcome, such as never treating a person merely as a means |
        | **virtue ethics** | asking what a person of good character, such as a just and honest one, would do |
        | **care ethics** | starting from relationships and from the people who depend on others, and listening to what they need |
        """),
        L.watch_out("Turning a theory into a number, as the exercise does with the veil of ignorance, makes one reading of it. Rawls did not write about hospital programs, and someone else can read him in another way: say which reading you use."),
        mo.Html(C.picture({"totals": "Readmissions prevented", "totals_y": "in total", "worst": "Worst-off group", "worst_y": "share helped",
                           "nearby": "nearby", "need": "need",
                           "alt": "Left, the readmissions each policy prevents in total: 32 for nearby first, 28 for need first. Right, the share helped in the group that is worst off: 0.02 for nearby first, 0.06 for need first."})),
        mo.md(r"""
        Left, what consequentialism adds up: nearby first wins, 32 against 28. Right, what someone behind the veil looks at: need first wins, 0.06 against 0.02. The same two policies, judged by two questions, come out the other way round.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To look at a decision through the theories:

    1. Write down the decision and the options.
    2. Consequences: what good and harm does each option do, and to whom, added up?
    3. Duties and rights: does an option break a rule or a right, whatever it brings about?
    4. A fair agreement: which rule would you choose without knowing who you will be?
    5. Character and care: what would a good, caring person do, and what do the people involved say they need?
    6. Where the answers differ, write down what is at stake.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The hospital of the guess. `people` says how many patients live in each group, `prevented` how many readmissions each policy prevents in each group. `prevented.items()` gives each policy with its own dictionary of groups, as in [Dictionaries](/course/open/py-dicts?lang=en). Run the cell.
    """)
    return


@app.cell
def example_hospital():
    _people = {"near": 300, "far": 100}
    _prevented = {"nearby first": {"near": 30, "far": 2}, "need first": {"near": 18, "far": 10}}

    # 2. consequences: the readmissions each policy prevents, added up over everyone
    _totals = {}
    for _policy, _groups in _prevented.items():
        _totals[_policy] = _groups["near"] + _groups["far"]
    print(_totals)
    # 4. a fair agreement: the share of each group that each policy helps
    for _policy, _groups in _prevented.items():
        print(_policy, _groups["near"] / _people["near"], _groups["far"] / _people["far"])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Nearby first prevents 32 readmissions, need first 28: step 2 picks nearby first. But nearby first helps 0.1 of the patients near the hospital and only 0.02 of those far away; need first helps 0.06 and 0.1. Behind the veil you do not know in which group you will be, so you look at the group that is worst off under each policy. Exercise 1 picks that group out.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_worst():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Behind the veil

        Replace the `...` with one line: for this policy, the share helped in the group that is worst off, which is the smaller of the two shares. `min(a, b)` gives the smaller of two numbers. The last line is `worst`, so the check sees it. The check also tries other numbers, in which the worst-off group changes.
        """),
        L.task("Put in `worst` the share helped in the worst-off group, under the name of each policy."),
    ])
    return


@app.cell(hide_code=True)
def exercise_worst():
    box_worst = L.code_box("worst", lines=7)
    box_worst
    return (box_worst,)


@app.cell(hide_code=True)
def check_worst(box_worst):
    result_worst = L.check("worst", box_worst.value)
    result_worst
    return (result_worst,)


@app.cell(hide_code=True)
def hints_worst():
    L.hints(
        "worst",
        [
            "Step 4 of the recipe: behind the veil you look, for each policy, at the group that is worst off.",
            "The worked example prints two shares for each policy. Which of the two belongs to the group that is worst off?",
            "`worst[policy] = min(groups[\"near\"] / ___, groups[\"far\"] / ___)`.",
        ],
        why="`groups[\"near\"] / people[\"near\"]` is the share helped near the hospital, and `groups[\"far\"] / people[\"far\"]` far away; `min` keeps the smaller one. For nearby first that is 0.02, far away; for need first 0.06, near the hospital.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_theories():
    mo.md(r"""
    ### Exercise 2. Which theory?

    Eight arguments about the hospital's choice. Put each one with the theory it comes from.
    """)
    return


@app.cell(hide_code=True)
def sort_theories_cell():
    sort_theories = L.sort("theories")
    sort_theories
    return (sort_theories,)


@app.cell(hide_code=True)
def check_theories(sort_theories):
    result_theories = L.check("theories", sort_theories)
    result_theories
    return (result_theories,)


@app.cell(hide_code=True)
def hints_theories():
    L.hints(
        "theories",
        [
            "Ask for each argument: does it add up outcomes, hold on to a rule or right, ask about character, or start from relationships and the people involved?",
            "*An equal right* holds whatever the total, and *A good doctor* is about character. Which other argument holds whatever the outcome, and which other one asks about character?",
            "Consequences: the total and \\_\\_\\_. Duties: an equal right and \\_\\_\\_. Virtues: a good doctor and \\_\\_\\_. Care: ask them and \\_\\_\\_.",
        ],
        why="**Consequences:** the total and later on. **Duties and rights:** an equal right and not a means. **Virtues:** a good doctor and what kind of hospital. **Care:** ask them and who depends on us.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_ubuntu():
    mo.md(r"""
    ### Exercise 3. What Ubuntu adds
    """)
    return


@app.cell(hide_code=True)
def choice_ubuntu_cell():
    choice_ubuntu = L.choice(
        "ubuntu",
        question="In Ubuntu, a person is a person through other people. Which question does it add to the hospital's choice?",
        options={
            "right": "What the choice does to the community and the ties within it",
            "rule": "Which rule a patient could accept without knowing where they will live",
            "total": "Which policy prevents the most readmissions",
        },
        correct="right",
        why={
            "rule": "That is the question of a fair agreement, Rawls's veil of ignorance. What does Ubuntu look at, if a person is a person through others? See *The idea*.",
            "total": "That is what consequentialism adds up. What does Ubuntu look at, if a person is a person through others? See *The idea*.",
            "right": "Ubuntu looks at the ties between people: a hospital that leaves out the villages far away also weakens its bond with those communities.",
        },
    )
    choice_ubuntu
    return (choice_ubuntu,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en) you learned to find harm; here you learned ways to judge it. The yes rate per group comes back as the share helped per group, and the veil of ignorance gave a reason to look at the group that is worst off. In the next lesson, *What should AI make better?*, you ask what a system should make better for people in the first place, and see that clicks and well-being are not the same.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_ubuntu, result_theories, result_worst):
    L.wrap_up(result_worst, result_theories, choice_ubuntu)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
