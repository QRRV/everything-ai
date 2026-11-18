# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-case-health-credit. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-case-health-credit", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [COMPAS and a hiring tool](/course/open/rai-case-compas-amazon?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_questions():
    quick_questions = L.choice(
        "quick-questions",
        question="ProPublica called the COMPAS scores biased, and Northpointe called them fair. How can both be true?",
        options={
            "right": "They asked other questions of the same scores",
            "data": "They each had different data about the defendants",
            "error": "One of them made a mistake in the arithmetic",
        },
        correct="right",
        why={
            "right": "Northpointe asked what share of the flagged people was arrested again; ProPublica asked what share of the people who were not arrested again was flagged anyway.",
            "data": "Both looked at the same scores from Broward County. See *The idea* in [COMPAS and a hiring tool](/course/open/rai-case-compas-amazon?lang=en).",
            "error": "Both sums were right. See *The idea* in [COMPAS and a hiring tool](/course/open/rai-case-compas-amazon?lang=en).",
        },
    )
    quick_questions
    return (quick_questions,)


@app.cell(hide_code=True)
def quick_check_amazon():
    quick_amazon = L.choice(
        "quick-amazon",
        question="Where did Amazon's model learn to mark down CVs with the word *women's*?",
        options={
            "rule": "From a rule that the team had written into it",
            "right": "From ten years of CVs, mostly of men",
            "ads": "From the words in the job ads that Amazon placed",
        },
        correct="right",
        why={
            "rule": "Nobody wrote such a rule; the team later tried to make the model ignore the word. See *The idea* in [COMPAS and a hiring tool](/course/open/rai-case-compas-amazon?lang=en).",
            "right": "It learned from patterns in the CVs sent to the company over ten years, and most of them came from men.",
            "ads": "It learned from the CVs, not from the job ads. See *The idea* in [COMPAS and a hiring tool](/course/open/rai-case-compas-amazon?lang=en).",
        },
    )
    quick_amazon
    return (quick_amazon,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In 2019 a study in *Science* found that an algorithm used for millions of patients in the United States, which helps hospitals choose who gets extra care, gave Black patients the same scores as White patients who were healthier. The algorithm did not use race. The problem was what it had learned to predict: not how ill someone was, but how much their care would cost. A model learns its label, whatever you meant by it.

    **An everyday example.** A school wants to find the students who need help with reading, and picks those whose parents bought the fewest books last year. Books bought are easy to count, but they measure money as much as reading.
    """)
    return


@app.cell(hide_code=True)
def predict_sicker():
    guess_sicker = L.guess(
        "sicker",
        question="At the scores where patients were picked for the program, how many more chronic illnesses do you think Black patients had than White patients?",
        options={
            "same": "About the same number of them",
            "few": "About 5 percent more",
            "right": "About 26 percent more",
        },
        reveal="**About 26 percent more**: 4.8 chronic illnesses against 3.8, at the score above which patients were picked automatically.",
    )
    guess_sicker
    return (guess_sicker,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **proxy label** is something you can measure that stands in for what you really want to know, as the costs of care stand in for the need for care. A model learns the proxy. Where the proxy and the real thing come apart, the model is wrong in a way that the usual numbers do not show.

        **The health algorithm.** Hospitals in the United States use commercial algorithms to pick patients with complex needs for a program of extra care. Ziad Obermeyer and colleagues studied one, typical of the industry, at one academic hospital, with 6,079 Black and 43,539 White patients from 2013 to 2015. The 3 percent of patients with the highest scores were picked for the program automatically. The algorithm left race out on purpose, and it predicted the costs of the next year about equally well for both groups. But the label was costs, not need: at the same number of chronic illnesses, 1,801 dollars a year less was spent on Black patients than on White patients, because of unequal access to care. So at the same score, Black patients were sicker. Of the patients picked automatically, 17.7 percent were Black. The researchers simulated a score that meant the same health in both groups: then it would have been 46.5 percent. A new label that combined health and costs took away 84 percent of the bias. The maker of the algorithm repeated the analysis on about 3.7 million patients across the country.

        **The Apple Card.** In November 2019 the software developer David Heinemeier Hansson wrote on Twitter that the Apple Card, issued by the bank Goldman Sachs, had given Heinemeier Hansson 20 times the credit limit of Heinemeier Hansson's wife. Steve Wozniak, a founder of Apple, wrote of far better terms than for Wozniak's wife, too. The bank said that gender played no part. The New York State Department of Financial Services looked at the data of nearly 400,000 applicants in New York, and reported in March 2021 that it found no evidence that women were treated worse, on purpose or in effect. Spouses often had different credit records, for example when only one of them was named on a mortgage. But customer service could not explain the limits to the people who complained, and the regulator wrote that this lack of transparency seemed to cause confusion that could have been avoided.

        In the health algorithm, the label measured something other than the question. In the Apple Card case, the regulator found no unlawful discrimination, but the people involved could not see why their limits differed, and so could not check it themselves.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **proxy label** | a label you can measure that stands in for what you really want to know, such as the costs of care instead of the need for care; the model learns the proxy, not the real goal |
        """),
        L.watch_out("A proxy label can look perfect by the usual numbers: this algorithm predicted costs about equally well for Black and White patients. Check the proxy against what you really want to know, for each group."),
        mo.Html(C.picture({"title": "What the cost label did to the patients picked automatically",
                           "groups": ["Black", "White"], "illnesses": "chronic illnesses",
                           "share": "Black patients picked (%)", "cost": "cost label", "simulated": "same health",
                           "decimal": ".",
                           "alt": "Two bar charts. Left: at the score above which patients were picked automatically, Black patients had 4.8 chronic illnesses and White patients 3.8. Right: of the patients picked automatically, 17.7 percent were Black with the cost label, and 46.5 percent would have been with a score that meant the same health in both groups."})),
        mo.md(r"""
        The figures of Obermeyer and colleagues. The open bar is their simulation, not something that happened.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To check the label of a model that decides about people:

    1. Write down what you really want to know, such as the need for care.
    2. Write down what the model learns instead: its label, such as the costs of care.
    3. Ask what else moves the label, apart from what you want to know, such as access to care.
    4. Compare, for each group, who the label picks with who the goal would pick.
    5. Check that the people a decision is about can learn why, so that they can check it too.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Eight patients, made up by the course: four Black patients, B1 to B4, and four White patients, W1 to W4. `_illnesses` holds the number of chronic illnesses of each patient, and `_costs` the costs of their care in the next year, in dollars. In this made-up data, a chronic illness means 2,000 dollars of care a year for a Black patient and 3,000 for a White patient. The loop goes through the patients with `_patient` and puts in `_picked` every patient with at least `_threshold` chronic illnesses: picking by need. Run the cell.
    """)
    return


@app.cell
def example_by_need():
    _illnesses = {"B1": 5, "B2": 4, "B3": 2, "B4": 1, "W1": 4, "W2": 3, "W3": 2, "W4": 1}
    _costs = {"B1": 10000, "B2": 8000, "B3": 4000, "B4": 2000,
              "W1": 12000, "W2": 9000, "W3": 6000, "W4": 3000}

    _threshold = 4
    _picked = []
    for _patient in _illnesses:
        if _illnesses[_patient] >= _threshold:
            _picked.append(_patient)
    print(_picked)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    By need, B1, B2 and W1 are picked. The program in the study did not pick by need, but by a score that learned costs. Step 4 of the recipe asks who the label picks: Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_picked():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Who does the cost label pick?

        The same eight patients, in `illnesses` and `costs`. This time the program picks every patient whose costs are at least `threshold` dollars, and `threshold` is `9000`. Replace the `...` with the lines of the worked example, with the names without their underscore, `picked` and `patient`, and pick by `costs` instead of by `illnesses`. The loop goes through `costs` with `patient`. Leave out the `print`: the last line is `picked`, so the check sees it. The check also tries another threshold.
        """),
        L.task("Make `picked` the list of the patients whose costs are at least `threshold`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_picked():
    box_picked = L.code_box("picked", lines=12)
    box_picked
    return (box_picked,)


@app.cell(hide_code=True)
def check_picked(box_picked):
    result_picked = L.check("picked", box_picked.value)
    result_picked
    return (result_picked,)


@app.cell(hide_code=True)
def hints_picked():
    L.hints(
        "picked",
        [
            "The worked example picks by illnesses. Do the same with `costs` and `threshold`, without the `print`.",
            "Start with an empty list. In the loop, an `if` compares the costs of `patient` with `threshold`. What goes into `picked` when the costs are high enough?",
            "`if costs[patient] >= threshold:` and under it `picked.___(___)`.",
        ],
        why="`picked = []`, then `for patient in costs:` with `if costs[patient] >= threshold:` and `picked.append(patient)` inside it. That gives `['B1', 'W1', 'W2']`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_proxies():
    mo.md(r"""
    ### Exercise 2. The goal or the proxy?

    Six things: two from this lesson, two from *COMPAS and a hiring tool*, and two about a news site, made up by the course. Put each one under what it is.
    """)
    return


@app.cell(hide_code=True)
def sort_proxies_cell():
    sort_proxies = L.sort("proxies")
    sort_proxies
    return (sort_proxies,)


@app.cell(hide_code=True)
def check_proxies(sort_proxies):
    result_proxies = L.check("proxies", sort_proxies)
    result_proxies
    return (result_proxies,)


@app.cell(hide_code=True)
def hints_proxies():
    L.hints(
        "proxies",
        [
            "Ask of each thing: is it what someone really wants to know, or what is easy to count in the records?",
            "The six come in pairs: one about patients, one about defendants and one about readers. In each pair, one stands in for the other.",
            "The goals: need for care, \\_\\_\\_ and interest. The proxies: costs, arrested again and \\_\\_\\_.",
        ],
        why="**What you really want to know:** need for care, a new crime and interest. **What gets measured instead:** costs, arrested again and clicks.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_regulator():
    mo.md(r"""
    ### Exercise 3. What did the regulator find?
    """)
    return


@app.cell(hide_code=True)
def choice_regulator_cell():
    choice_regulator = L.choice(
        "regulator",
        question="What did the New York regulator report about the Apple Card in March 2021?",
        options={
            "women": "That the bank gave women lower limits on purpose, by a rule",
            "right": "No unlawful discrimination, but too little explanation",
            "fine": "Nothing was wrong, the customer service included",
        },
        correct="right",
        why={
            "women": "The regulator found no rule that gave women lower limits, and no evidence of discrimination on purpose or in effect. See *The idea*.",
            "right": "It found no evidence that women were treated worse, but customer service could not explain the limits to the people who complained, and they were left confused.",
            "fine": "The regulator found no unlawful discrimination, but it did find fault. Could the customer service explain the limits? See *The idea*.",
        },
    )
    choice_regulator
    return (choice_regulator,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [COMPAS and a hiring tool](/course/open/rai-case-compas-amazon?lang=en) you saw that harm can come from the data and from what was measured, and that COMPAS learned arrests, not crimes. In the health algorithm the label itself was the problem: costs stood in for need, and the usual numbers looked fine. In the Apple Card case the regulator found no unlawful discrimination, but people could not see why their limits differed. In the next lesson, *Labels, a chatbot and microtargeting*, you look at harm in what a system says and in what it does with data.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_regulator, result_picked, result_proxies):
    L.wrap_up(result_picked, result_proxies, choice_regulator)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
