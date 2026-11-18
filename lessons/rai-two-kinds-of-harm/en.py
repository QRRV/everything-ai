# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-two-kinds-of-harm. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-two-kinds-of-harm", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Who is in the data, and who is missing?](/course/open/rai-bias-in-data?lang=en)
    """)
    return


@app.cell(hide_code=True)
def quick_check_stakeholder():
    quick_stakeholder = L.choice(
        "quick-stakeholder",
        question="A city decides which streets to repair first with a model that reads complaints sent through an app. Who are stakeholders of that model?",
        options={
            "app": "Only the people who send complaints",
            "city": "Only the city workers who plan the repairs, since they are the ones who use it",
            "right": "Also the residents who never use the app",
        },
        correct="right",
        why={
            "app": "A stakeholder does not have to use the system. What happens to a street where nobody uses the app? See *The idea* in [Who is in the data, and who is missing?](/course/open/rai-bias-in-data?lang=en)",
            "city": "The people who use a system are only some of its stakeholders. Who lives with the streets it puts last? See *The idea* in [Who is in the data, and who is missing?](/course/open/rai-bias-in-data?lang=en)",
            "right": "Their streets are judged too, from complaints they never send, so the model can put them last. The people who use the app and the city have a stake as well.",
        },
    )
    quick_stakeholder
    return (quick_stakeholder,)


@app.cell(hide_code=True)
def quick_check_score():
    quick_score = L.choice(
        "quick-score",
        question="A model is right for 92% of a test set, and for 70% of group B, which is 8% of that test set. What do you write next to the score?",
        options={
            "right": "92% overall, but 70% for group B, 8% of the test set",
            "drop": "Nothing more: the 92% already counts group B, so it says it all",
            "mean": "The mean of the two, 81%, as a fairer overall score",
        },
        correct="right",
        why={
            "right": "Step 6 of the recipe in that lesson: the score per group next to the overall one, and how big the group is, so that a reader sees for whom it holds.",
            "drop": "Group B is counted in the 92%, but it weighs only 8%. What does the overall score hide? See the formula in [Who is in the data, and who is missing?](/course/open/rai-bias-in-data?lang=en)",
            "mean": "A mean of two scores is a new number that holds for nobody. What would a reader need to see? See the recipe in [Who is in the data, and who is missing?](/course/open/rai-bias-in-data?lang=en)",
        },
    )
    quick_score
    return (quick_score,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model that turns down a loan does harm you can point at: a person did not get something. But in a study published in 2015, Matthew Kay, Cynthia Matuszek and Sean Munson searched for images of people at work, job by job, and compared who the pictures showed with who really does those jobs. Nobody was turned down for anything in that search, and still something was wrong with it. This lesson is about both kinds of harm, and about how you find each one.

    **An everyday example.** A school book in which every doctor is drawn as a man refuses no one a place at medical school. It still tells every child who reads it who becomes a doctor.
    """)
    return


@app.cell(hide_code=True)
def predict_ceo():
    guess_ceo = L.guess(
        "ceo",
        question="According to the US labor statistics in that study, 27% of the CEOs in the US were women. What share of the image search results for CEO do you think showed a woman?",
        options={
            "same": "About 27%: a search shows the world as it is",
            "more": "More than 27%: search engines try to show variety",
            "right": "11%",
        },
        reveal="**11%.** That is less than half of the share among real CEOs. The search turned nobody down, but it showed a world that is more one-sided than the real one, and a picture like that can make a stereotype stronger.",
    )
    guess_ceo
    return (guess_ceo,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        In 2017, Solon Barocas, Kate Crawford and colleagues gave the two kinds their names, and Crawford's talk at NIPS that year (now called NeurIPS) made them widely known.

        - **Allocative harm** comes from what a system hands out: a loan, a job, a place at a school, but also a burden such as an extra check. It is harm when some groups get less of the good, or more of the bad, than they should. You can measure it: per group, the share of people that gets a yes.
        - **Representational harm** comes from how a system shows or describes people: as a stereotype, with an insult, or not at all. Crawford also counted a system that recognizes some groups worse, such as the face detector of the previous lesson; later lists often give that its own name, a quality-of-service harm, and you measure it per group, as the previous lesson did. Stereotypes, insults and groups left out decide nothing about anyone, so there is no yes to count. You have to look for them: collect many outputs, write down who is shown and how, and count.

        Both kinds are about **fairness**, the aim that a system does not treat people or groups unjustly. For what a system hands out, the yes rate of a group is:
        """),
        L.formula(r"\text{yes rate of a group} = \frac{\text{yes in the group}}{\text{people in the group}}"),
        mo.md(r"""
        A gap between the yes rates of two groups does not prove that a system is unfair: the groups can differ in ways that matter for the decision. It tells you where to look. Later lessons ask when such a gap is unjust.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **allocative harm** | harm from what a system hands out, such as loans, jobs or extra checks, when some groups get less of the good or more of the bad than they should |
        | **representational harm** | harm from how a system shows or describes groups: stereotypes, insults, or a group left out; it decides nothing about anyone |
        | **fairness** | the aim that a system does not treat people or groups unjustly |
        """),
        L.watch_out("Stereotypes, insults and groups left out do not show up in a score. Nobody is turned down, so there is no yes rate to compare: if you only measure decisions, you never see them."),
        mo.Html(C.picture({"loans": "Yes for a loan", "loans_y": "yes rate", "ceo": "Women among CEOs", "ceo_y": "share",
                           "images": "images", "real": "real",
                           "alt": "Left, the yes rate of a loan model per group: 0.75 for group A, 0.375 for group B. Right, the share of women: 0.11 in the image search results for CEO in a study published in 2015, 0.27 among real CEOs in the US."})),
        mo.md(r"""
        Left, allocative: a loan model that says yes to 0.75 of the applications of group A and to 0.375 of those of group B. These numbers are made up by the course. Right, representational: 0.11 of the images for CEO showed a woman, against 0.27 of the real CEOs. The left gap is about who gets something, the right one about how a group is shown.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find both kinds of harm:

    1. Ask what the system hands out (a yes, a loan, a check), and what it shows or writes about people.
    2. For what it hands out: per group, divide the number of yeses by the number of people.
    3. For what it shows: ask for many outputs, and write down for each one who is shown, and how.
    4. Count per group or role, and compare with something to hold it against, such as the share in the real world.
    5. Write down what you found, next to the score.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The loan model of the picture: 20 applications, made up by the course, 1 in `yes` for a loan it grants and 0 for one it turns down. The loop puts the yes rate of each group in a dictionary, under the name of the group, as `rates[group] = ...` does. Run the cell.
    """)
    return


@app.cell
def example_loans():
    import pandas as pd

    # 1. what it hands out: a yes or a no for a loan
    _groups = ["A"] * 12 + ["B"] * 8
    _yes = [1] * 9 + [0] * 3 + [1] * 3 + [0] * 5
    _loans = pd.DataFrame({"group": _groups, "yes": _yes})
    # 2. per group, the yeses divided by the people: the mean of `yes`
    _rates = {}
    for _group in ["A", "B"]:
        _rates[_group] = _loans[_loans["group"] == _group]["yes"].mean()
        print(_group, _rates[_group])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Group A gets a yes for 9 of its 12 applications, 0.75; group B for 3 of its 8, 0.375: half as often. Step 5 is a sentence next to the score: "says yes to 75% of the applications of group A and to 37.5% of those of group B". Steps 3 and 4 are not in this code, because a loan is handed out, not shown. Exercise 1 does them.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shares():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Who is in the pictures?

        An image generator was asked ten times each for a picture of a nurse, an engineer, a teacher and a CEO. Someone looked at the 40 pictures and wrote down for each one whether it shows a woman: 1 if it does, 0 if not. Judging that from a picture is itself a rough label, as the question of who gave the labels in the previous lesson warns. The pictures, and the shares of women in each job that the check compares them with, are made up for this exercise. Replace the `...` with one line, as in the worked example. The last line is `shares`, so the check sees it. The check also tries other pictures.
        """),
        L.task("Put in `shares` the share of women among the pictures of each job, under the name of the job."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shares():
    box_shares = L.code_box("shares", lines=11)
    box_shares
    return (box_shares,)


@app.cell(hide_code=True)
def check_shares(box_shares):
    result_shares = L.check("shares", box_shares.value)
    result_shares
    return (result_shares,)


@app.cell(hide_code=True)
def hints_shares():
    L.hints(
        "shares",
        [
            "Steps 3 and 4 of the recipe, as step 2 in the worked example: per job, the share of its pictures that show a woman.",
            "Which mask keeps only the rows of one job? And how does a value go into a dictionary under a key?",
            "`shares[job] = images[___][\"woman\"].___()`.",
        ],
        why="`images[images[\"job\"] == job]` keeps the ten pictures of one job, and the mean of their `woman` is the share that shows a woman: 10 of 10 for nurse, 1 for engineer, 9 for teacher and 1 for CEO. In every job the pictures are more one-sided than the made-up workforce, in the direction of the stereotype.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Exercise 2. Which kind of harm?

    Six cases. Put each one with the kind of harm it does.
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
            "Ask for each case: does someone get something here, or not get it? Or is it about how a group is shown or described?",
            "Three cases decide something about a person. Which three decide nothing about anyone?",
            "Allocative: the loan, \\_\\_\\_ and the job ad. Representational: \\_\\_\\_, autocomplete and \\_\\_\\_.",
        ],
        why="**Allocative:** the loan, the extra checks and the job ad. Each is handed to some people and not to others, so you can count per group who gets it. **Representational:** the translation, autocomplete and the CEOs. They decide nothing about anyone, but they show a group as less, or as a stereotype.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_look_for():
    mo.md(r"""
    ### Exercise 3. Looking for it
    """)
    return


@app.cell(hide_code=True)
def choice_look_for_cell():
    choice_look_for = L.choice(
        "look-for",
        question="A company builds a chatbot that writes short stories for children. How do you look for representational harm before it goes live?",
        options={
            "complaints": "Wait for complaints from parents once it is live, and fix what they find",
            "right": "Ask it for many stories and count who is described how",
            "score": "Measure its accuracy per group of readers",
        },
        correct="right",
        why={
            "complaints": "By then the stories have reached children, and harm that nobody looks for is rarely reported. What can you do before it goes live? See the recipe.",
            "right": "Steps 3 and 4 of the recipe: ask for many stories, write down who appears in which role, such as the hero, the doctor or the villain, and count per group, as in Exercise 1.",
            "score": "A story has no right answer to score it against, and a stereotype is not a wrong answer. What would you count? See *The idea*.",
        },
    )
    choice_look_for
    return (choice_look_for,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Who is in the data, and who is missing?](/course/open/rai-bias-in-data?lang=en) you measured a score per group. Here the same count, per group, measures allocative harm, and counting what a model shows is how you look for representational harm. In the next lesson, *Ethical theories applied to AI*, you meet ways to judge whether a harm like this is wrong, and why.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_look_for, result_kinds, result_shares):
    L.wrap_up(result_shares, result_kinds, choice_look_for)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
