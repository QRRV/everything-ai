# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-human-oversight. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-human-oversight", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Who is responsible?](/course/open/rai-accountability?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_crumple():
    quick_crumple = L.choice(
        "quick-crumple",
        question="A test car's safety driver is blamed for a crash that the car's design made likely. What is that called?",
        options={
            "hands": "The problem of many hands",
            "right": "A moral crumple zone",
            "gap": "A responsibility gap: nobody could foresee it",
        },
        correct="right",
        why={
            "hands": "With many hands, the blame spreads until nobody holds it. Here it lands on one person. See *The idea* in [Who is responsible?](/course/open/rai-accountability?lang=en).",
            "right": "The person nearest the machine absorbs the blame, as the crumple zone of a car absorbs a crash.",
            "gap": "Here somebody is blamed: the driver. See *The idea* in [Who is responsible?](/course/open/rai-accountability?lang=en).",
        },
    )
    quick_crumple
    return (quick_crumple,)


@app.cell(hide_code=True)
def quick_check_overlap():
    quick_overlap = L.choice(
        "quick-overlap",
        question="Two teams both say they answer for testing the model. Why is that a problem?",
        options={
            "right": "Each may count on the other to do it",
            "slow": "Testing twice takes twice as long, which delays the launch",
            "none": "It is not: two names are twice as safe",
        },
        correct="right",
        why={
            "right": "An overlap can leave a gap as wide as no name at all: give the step one owner.",
            "slow": "The worry is not the time, but that nobody tests at all. See the *Watch out* box in [Who is responsible?](/course/open/rai-accountability?lang=en).",
            "none": "Two names are not twice as safe. See the *Watch out* box in [Who is responsible?](/course/open/rai-accountability?lang=en).",
        },
    )
    quick_overlap
    return (quick_overlap,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A common answer to *who is responsible?* is: a person checks what the system decides. European law asks for it, and a company that uses AI often says it to reassure people. But a person who checks hundreds of the system's suggestions a day, and finds them right nearly every time, soon stops looking closely. Then the check is a signature, and the person is the crumple zone of the previous lesson. This lesson shows when a person in the loop helps, and how to find out.

    **An everyday example.** A driver follows the navigation into a road that is closed, past a sign that says so: the screen has been right a hundred times before.
    """)
    return


@app.cell(hide_code=True)
def predict_missed():
    guess_missed = L.guess(
        "missed",
        question="Students flew a simulator with an aid that was usually right. When the aid stayed silent about an event their own instruments showed, how often do you think they missed it?",
        options={
            "few": "About 5 percent of the time",
            "right": "About 40 percent",
            "most": "About 80 percent",
        },
        reveal="**About 40 percent**: 41 percent, against 3 percent for students who flew without the aid. And when the aid gave a wrong instruction that their instruments contradicted, the error rate was about 65 percent (Skitka, Mosier and Burdick, 1999).",
    )
    guess_missed
    return (guess_missed,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Human oversight** means that a person watches what an AI system does and can step in. The European privacy law, the GDPR, gives people the right not to be subject to a decision about them that is made fully automatically and has legal or similarly serious effects. Where such a decision is allowed, people have at least the right to have a person look at it, to give their view and to contest it. The AI Act asks that high-risk systems be built so that people can oversee them: understand what the system can and cannot do, stay aware that they may rely on it too much, read its output correctly, decide not to use it or to override it, and stop it.

        The second point has a name. **Automation bias** is relying on a system's suggestion more than the evidence allows. It shows up in two ways: missing a problem because the system did not point at it, and following a wrong suggestion although other information says it is wrong.

        Oversight works only when the person has three things: *time* to look at each case, *information* about the case itself and not only the system's score, and the *power* to say no without paying for it. And it works only if you check it. How often the person agrees with the system tells you little; what counts is how many of the system's mistakes the person catches. In 2022 the researcher Ben Green read 41 policies that require human oversight of government algorithms, and found that people often cannot do what those policies expect, while the oversight makes the system look safe. He argues for checking the whole system and its use, not only adding a person.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **human oversight** | a person monitoring an AI system and able to step in: to understand its output, overrule a decision or stop it |
        | **automation bias** | relying on an automated system's suggestion more than the evidence allows: missing what it did not flag, and following its mistakes |
        """),
        L.watch_out("A reviewer who agrees with the system 9 times out of 10 is not proof that the system is good, or that the review works. Look at the cases where the system was wrong: what did the reviewer do there?"),
        mo.Html(C.picture({"title": "The model's mistakes and the reviewer's changes",
                           "wrong": "model wrong", "changed": "reviewer changed it", "xlabel": "application",
                           "alt": "Twenty loan applications in a row. The model was wrong on applications 4, 7, 11 and 16. The reviewer changed the model's decision on applications 11 and 14: only one of the four mistakes was put right, and the change on 14 made a right decision wrong."})),
        mo.md(r"""
        A bank made up by the course. A filled dot on top means that the model's suggestion was wrong, as an audit found later; a filled dot below means that the reviewer changed it.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To set up a person in the loop:

    1. Write down which decisions a person checks, and what the person can do: confirm, change or stop.
    2. Give the person time, the case itself and not only a score, and the power to say no without paying for it.
    3. Count how often the person agrees with the system.
    4. On a sample where you know the right answer, count the system's mistakes, and how many of them the person put right.
    5. When the person catches few mistakes, change the setup, not only the person.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Five applications, made up by the course: 1 is yes, 0 is no. Step 3 of the recipe: count how often the reviewer agreed with the model. Run the cell.
    """)
    return


@app.cell
def example_agreed():
    _model = [1, 0, 1, 1, 0]
    _person = [1, 0, 1, 1, 1]

    _agreed = 0
    for _i in range(len(_model)):
        if _person[_i] == _model[_i]:
            _agreed = _agreed + 1
    print(_agreed, "of", len(_model))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The reviewer agreed 4 times out of 5. That says how often the reviewer said yes to the model, not whether either of them was right. For that you need the right answers, from an audit, and step 4 of the recipe: Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_caught():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Which mistakes were caught?

        The bank of the picture. `model` holds the model's suggestions, `person` the reviewer's decisions and `right` what the audit found, with 1 for yes and 0 for no. The loop already counts the model's mistakes in `mistakes`. Replace the `...` with two lines: an `if` about the reviewer's decision, and under it a line that adds 1 to `caught`. The last line is `caught`, so the check sees it. The check also tries another reviewer.
        """),
        L.task("Make `caught` count the model's mistakes that the reviewer put right."),
    ])
    return


@app.cell(hide_code=True)
def exercise_caught():
    box_caught = L.code_box("caught", lines=12)
    box_caught
    return (box_caught,)


@app.cell(hide_code=True)
def check_caught(box_caught):
    result_caught = L.check("caught", box_caught.value)
    result_caught
    return (result_caught,)


@app.cell(hide_code=True)
def hints_caught():
    L.hints(
        "caught",
        [
            "You are already inside the `if` for the model's mistakes. There, one more `if` asks about the reviewer.",
            "Which two lists do you compare to know that the reviewer got it right? And what happens to `caught` then?",
            "`if person[i] == ___:` and under it `caught = caught + 1`",
        ],
        why="`if person[i] == right[i]:` with `caught = caught + 1` under it gives `1`: of the model's 4 mistakes, the reviewer put right only the one on application 11.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_missing():
    mo.md(r"""
    ### Exercise 2. What is the reviewer missing?

    Six setups, made up by the course, where a person checks a system. Put each one with what the person is missing.
    """)
    return


@app.cell(hide_code=True)
def sort_missing_cell():
    sort_missing = L.sort("missing")
    sort_missing
    return (sort_missing,)


@app.cell(hide_code=True)
def check_missing(sort_missing):
    result_missing = L.check("missing", sort_missing)
    result_missing
    return (result_missing,)


@app.cell(hide_code=True)
def hints_missing():
    L.hints(
        "missing",
        [
            "Ask for each setup: could the reviewer look closely, could the reviewer see a mistake, and would finding one change anything?",
            "Two setups are about speed. In two others, the reviewer can see something wrong, but cannot act on it or loses by acting.",
            "Time: a minute each and \\_\\_\\_. Information: only a score and \\_\\_\\_. Power: \\_\\_\\_ and \\_\\_\\_.",
        ],
        why="**Not enough time:** a minute each and every few seconds. **Not enough information:** only a score and no reason. **No power to say no:** a bonus for agreeing and already sent.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_audit():
    mo.md(r"""
    ### Exercise 3. What to ask first
    """)
    return


@app.cell(hide_code=True)
def choice_audit_cell():
    choice_audit = L.choice(
        "audit",
        question="A town says its fraud model is safe, because a person checks every case it flags. What do you ask first?",
        options={
            "count": "How many people check the flags, and how many flags each one handles in a day",
            "law": "Whether every reviewer has had a course about the law",
            "right": "How often a reviewer overrules a flag, and whether rightly",
        },
        correct="right",
        why={
            "count": "That tells you something about time, one of the three conditions, but not whether the check works. See *The idea*.",
            "law": "A course may help, but it does not show that reviewers catch the model's mistakes. See *The idea*.",
            "right": "Only what the reviewers do when the model is wrong shows whether the check works; a check that never overrules is a signature.",
        },
    )
    choice_audit
    return (choice_audit,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Who is responsible?](/course/open/rai-accountability?lang=en) you saw that the blame tends to land on the person nearest the machine. Here you saw why that person often cannot do much: automation bias, and a setup without time, information or power. In the next lesson, *Deciding about people automatically*, you ask when a decision about people may be automated at all, and what someone needs to be able to object.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_audit, result_caught, result_missing):
    L.wrap_up(result_caught, result_missing, choice_audit)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
