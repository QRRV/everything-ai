# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-reading-claims. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-reading-claims", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [What a paper, a preprint and peer review are](/course/open/res-what-is-a-paper?lang=en) and [Always beat a simple baseline](/course/open/ml-baselines?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_strong():
    quick_strong = L.choice(
        "quick-strong",
        question="Which of these supports a result most strongly?",
        options={
            "reviewed": "A paper about it that passed peer review",
            "press": "A press release from the lab that found it, with quotes from the head of the lab",
            "right": "The same result, found again by other researchers",
        },
        correct="right",
        why={
            "reviewed": "Peer review helps, but reviewers rarely run the experiments again. See [What a paper, a preprint and peer review are](/course/open/res-what-is-a-paper?lang=en).",
            "press": "A press release is the lab's own summary, not a check by anyone else. See [What a paper, a preprint and peer review are](/course/open/res-what-is-a-paper?lang=en).",
            "right": "A result that held up in other hands is the strongest: that is the last step from preprint to solid result.",
        },
    )
    quick_strong
    return (quick_strong,)


@app.cell(hide_code=True)
def quick_check_baseline():
    quick_baseline = L.choice(
        "quick-baseline",
        question="A model gets an accuracy of 0.85 on its test set. Always predicting the most common class gets 0.80, measured on the training set. What can you say?",
        options={
            "right": "Nothing yet: measure the baseline on the same test set",
            "beats": "The model beats its baseline: 0.85 is higher than 0.80",
            "close": "The model learned little: 0.85 is close to 0.80",
        },
        correct="right",
        why={
            "beats": "The two numbers were measured on different sets, so you cannot compare them. See [Always beat a simple baseline](/course/open/ml-baselines?lang=en).",
            "close": "Whether the gap is small, you only know once both are measured on the same test set. See [Always beat a simple baseline](/course/open/ml-baselines?lang=en).",
            "right": "A comparison counts only when both are measured on the same test set, with the same metric.",
        },
    )
    quick_baseline
    return (quick_baseline,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    "Our AI beats doctors." "The new model understands mathematics." Claims like these reach you through press releases, news stories and posts, and each step makes them shorter and surer. Behind them there is usually one number on one test. To know what a claim is worth, you trace it back to what was actually measured.

    **An everyday example.** An advert says a toothpaste makes teeth "up to 40% whiter". The small print says: in one test, with twelve people, after eight weeks, compared with no brushing at all.
    """)
    return


@app.cell(hide_code=True)
def predict_new_set():
    guess_new_set = L.guess(
        "new-set",
        question="Many language models score high on a well-known set of maths word problems. Someone writes a new set from scratch, with problems of the same kind and just as hard. How do you think the models score on the new set?",
        options={
            "same": "About the same: the problems are just as hard",
            "higher": "Higher: they have practised on this kind of problem",
            "right": "Some clearly lower: they may know the old questions",
        },
        reveal="**Some clearly lower.** When researchers did this in 2024 with a popular set of school maths problems, several models scored lower on the new problems, by up to 8 percentage points, while others hardly dropped. The ones that dropped had probably seen the old questions, or questions much like them, while they were trained: the questions had been on the internet for years. A score on such a set shows how a model does on those questions, not on every question of that kind.",
    )
    guess_new_set
    return (guess_new_set,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **benchmark** is a fixed set of test tasks with a fixed way of scoring them, so that different models can be measured the same way and their scores compared. Most claims about AI come down to a score on a benchmark, or on a test set of the researchers' own.

        Three kinds of text say different things about the same result:

        - **The benchmark score**: one number, on those tasks, scored that way. That is what was measured.
        - **The paper**: how the score was measured, on which data, against what, and with which limits. The limits are often in a section near the end.
        - **The press release or news story**: what the result is supposed to mean. It is the shortest and the surest of the three, and the furthest from the measurement.

        So ask, about every claim: **what exactly was measured, and against what?** A score means something only next to a comparison: a baseline as in [Always beat a simple baseline](/course/open/ml-baselines?lang=en), earlier models on the same benchmark, or people under the same conditions. And then: **what does the claim say that the measurement does not show?** That gap is where a claim gets too big.

        A benchmark also wears out. Its questions end up on the internet, and so in the training data of later models. A model can then score high because it has seen the questions, not because it can do the task: that is called contamination, and a later lesson shows how model makers try to prevent it. A new set of similar questions shows the difference.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **benchmark** | | a fixed set of test tasks with a fixed way of scoring, so that models can be compared | |
        """),
        L.watch_out("A model with a higher score on a benchmark is not \"better\" in general: it is better on those tasks, scored that way. Whether that carries over to what you want to use the model for, the benchmark does not tell you."),
        mo.md(r"""
        The picture: how a claim grows from the measurement to the headline, with the example of the worked example below.
        """),
        mo.Html(C.ladder({"title": "From measurement to headline",
                          "rows": [("Measured", "86% right on 200 photos from one hospital"),
                                   ("Paper", "higher than 16 skin doctors who saw only the same photos"),
                                   ("Press release", "\"Our AI diagnoses skin cancer better than doctors\"")],
                          "alt": "Three boxes from bottom to top. Measured: 86% right on 200 photos from one hospital. Paper: higher than 16 skin doctors who saw only the same photos. Press release: our AI diagnoses skin cancer better than doctors. Each box is wider than the one below it."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To check a claim about AI:

    1. Write down exactly what is claimed, in one sentence.
    2. Find what was measured: which task or benchmark, which test data, and which number.
    3. Find what it was compared with: a baseline, earlier models, or people, and under which conditions.
    4. Put the claim next to the measurement: what does the claim say that the measurement does not show?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A made-up example, built like many real ones. A press release says: **"Our AI diagnoses skin cancer better than doctors."** The paper behind it says: the model judged 200 photos of skin spots, all from one hospital, and was right for 86% of them. Half of the photos showed skin cancer, so always saying "no cancer" would be right for 50%. Sixteen skin doctors judged the same 200 photos, with nothing else to go on, and were right for 79% on average.

    1. **The claim:** the AI diagnoses skin cancer better than doctors.
    2. **Measured:** the accuracy on 200 photos from one hospital: 86% for the model.
    3. **Compared with:** the baseline of always saying "no cancer", 50%, and 16 skin doctors, 79% on average, who saw only the photos. In their practice they also see the patient, ask questions and can touch the spot.
    4. **Claim next to measurement:** "diagnoses" and "better than doctors" go further than "more right answers on 200 photos, against doctors who only had the photos". Whether it holds for photos from other hospitals, other cameras or other skin colours, this test does not show.

    The measurement is real and interesting. The claim is bigger than the measurement.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_measured():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Measured, or more?

        Six statements about a model that was tested on maths questions. Put each in its group.
        """),
    ])
    return


@app.cell(hide_code=True)
def exercise_measured():
    sort_measured = L.sort("measured")
    sort_measured
    return (sort_measured,)


@app.cell(hide_code=True)
def check_measured(sort_measured):
    result_measured = L.check("measured", sort_measured)
    result_measured
    return (result_measured,)


@app.cell(hide_code=True)
def hints_measured():
    L.hints(
        "measured",
        [
            "For each statement, ask: does it say what happened on a test, or more than that test could show?",
            "Look back at step 4 of the worked example: \"more right answers on 200 photos\" was measured; \"diagnoses\" and \"better than doctors\" went further. Which statements are like the first, and which like the second?",
            "Three are measurements: a score on a set of questions, or a comparison of scores on the same questions. The other three say what the model understands, what it will do, or how it does on questions it was never tested on.",
        ],
        why="**Measured:** 92% of 1,000 test questions right; 4 points higher than last year's best model on the same benchmark; 9 points lower on new similar questions than on the benchmark. **Goes beyond:** understands mathematics; will replace maths teachers; 92% of all school maths problems, when it was tested on one set of them.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_contamination():
    mo.md(r"""
    ### Exercise 2. A score that jumped
    """)
    return


@app.cell(hide_code=True)
def choice_contamination_cell():
    choice_contamination = L.choice(
        "contamination",
        question="A new language model scores 95% on a famous benchmark, where last year's best model scored 70%. The benchmark's questions and answers have been on the internet for years. What do you ask first?",
        options={
            "size": "How many parameters the new model has",
            "right": "Whether the benchmark's questions were in its training data",
            "press": "Whether the press release calls it a breakthrough",
        },
        correct="right",
        why={
            "size": "Size can explain a better model, but not whether the score measures the task. See contamination in *The idea*.",
            "press": "The press release is the furthest from the measurement. See *The idea*.",
            "right": "If the model saw the questions while training, the score shows memory, not skill: contamination. A new set of similar questions would tell.",
        },
    )
    choice_contamination
    return (choice_contamination,)


@app.cell(hide_code=True)
def text_exercise_claim():
    mo.md(r"""
    ### Exercise 3. Check a claim yourself

    Another made-up example. A news story: **"AI now writes better code than programmers."** The paper behind it: on a benchmark of 164 short programming tasks, each with tests, the model's code passed the tests for 90% of the tasks. The paper compares it with earlier models on the same benchmark, the best of which got 81%. It does not test programmers.
    """)
    return


@app.cell(hide_code=True)
def explain_claim():
    explain_check = L.explain(
        "check-claim",
        question="Check the claim with the four steps of the recipe. Write down what was measured, what it was compared with, and what the claim says that the measurement does not show.",
        model_answer="**Claim:** AI writes better code than programmers. **Measured:** passing the tests for 90% of 164 short tasks of one benchmark. **Compared with:** earlier models on the same benchmark, at most 81%; not with programmers at all. **Claim next to measurement:** \"better than programmers\" was never measured, and short tasks with ready tests are not the code programmers write every day.",
        points=[
            "you name the measurement: 90% of 164 short tasks of one benchmark",
            "you name the comparison: earlier models, not programmers",
            "you say that \"better than programmers\" goes beyond what was measured",
        ],
    )
    explain_check
    return (explain_check,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [What a paper, a preprint and peer review are](/course/open/res-what-is-a-paper?lang=en) showed how far a paper has come; now you can see how far a claim strays from what its paper measured. As in [Always beat a simple baseline](/course/open/ml-baselines?lang=en), a score means something only next to a comparison. In the next lesson, *Reading an abstract, a figure and a results table*, you read the parts of a paper where the measurements are.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_contamination, explain_check, result_measured):
    L.wrap_up(result_measured, choice_contamination, explain_check)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
