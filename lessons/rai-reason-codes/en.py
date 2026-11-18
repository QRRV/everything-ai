# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-reason-codes. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-reason-codes", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Kinds of explanation](/course/open/rai-kinds-of-explanation?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kind():
    quick_kind = L.choice(
        "quick-kind",
        question="For one applicant, you multiply each weight of a logistic regression by the applicant's value. What kind of explanation is that?",
        options={
            "global": "A global one, read from the model",
            "right": "A local one, read from the model",
            "after": "A local one, made afterwards",
        },
        correct="right",
        why={
            "global": "The weights alone are global, the same for everyone. Weight times value is about one applicant: a local explanation, read straight from the model. See [Kinds of explanation](/course/open/rai-kinds-of-explanation?lang=en).",
            "right": "It is about one applicant, so local, and it uses the model's own weights, so it comes from the model itself: nothing is made afterwards.",
            "after": "It is local, but nothing is made afterwards: you read the model's own weights. A post-hoc explanation is a separate method that studies a finished model. See [Kinds of explanation](/course/open/rai-kinds-of-explanation?lang=en).",
        },
    )
    quick_kind
    return (quick_kind,)


@app.cell(hide_code=True)
def quick_check_add():
    quick_add = L.choice(
        "quick-add",
        question="In the loan model of the previous lesson, `missed_payments` has the weight −1.2. Applicant A missed 2 payments. What does `missed_payments` add to A's logit?",
        options={
            "right": "−2.4",
            "weight": "−1.2, its weight",
            "value": "2, A's value",
        },
        correct="right",
        why={
            "right": "Weight times value: −1.2 · 2 = −2.4. Of A's three features, it weighed most against the yes.",
            "weight": "The weight is the same for every applicant. What it adds for A is the weight times A's value: −1.2 · 2 = −2.4. See [Kinds of explanation](/course/open/rai-kinds-of-explanation?lang=en).",
            "value": "The value alone says nothing about the model. What a feature adds is its weight times the value: −1.2 · 2 = −2.4. See [Kinds of explanation](/course/open/rai-kinds-of-explanation?lang=en).",
        },
    )
    quick_add
    return (quick_add,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Suppose a bank turns down your loan. In the United States it then has to tell you why. Regulation B, the rules under the Equal Credit Opportunity Act, asks for "a statement of specific reasons", which "must be specific and indicate the principal reason(s)". That you "failed to achieve a qualifying score on the creditor's credit scoring system" is not enough: the regulation calls it "insufficient". In the European Union, the GDPR gives a person about whom a decision with legal or similarly significant effects was made solely by automated means a right to "meaningful information about the logic involved", and in 2025 the Court of Justice of the EU called that "a genuine right to an explanation". A model that decides about a loan, then, has to come with a few reasons a person can follow. That is a local explanation, turned into words.

    **An everyday example.** A test comes back with a fail. That tells you what, not why. "You lost the most points on fractions, then on percentages" tells you what to practice, and it measures you against a pass: why this result, and not the other one.
    """)
    return


@app.cell(hide_code=True)
def predict_count():
    guess_count = L.guess(
        "count",
        question="How many reasons does Regulation B require a lender to give with a no?",
        options={
            "four": "Exactly four",
            "most": "At most four",
            "none": "No fixed number",
        },
        reveal="**No fixed number.** The official comment says the regulation \"does not mandate that a specific number of reasons be disclosed, but disclosure of more than four reasons is not likely to be helpful to the applicant.\" A hard limit of four comes from another law, the Fair Credit Reporting Act: with a credit score you get at most four \"key factors\", five when the number of inquiries about you is one of them. And those key factors do not count as the specific reasons Regulation B asks for.",
    )
    guess_count
    return (guess_count,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Reasons from a model.** In practice a lender sends its reasons as short, fixed phrases, each with a code: reason codes. That name is industry usage; the law speaks of "specific reasons" and "principal reason(s)". The reasons have to be the real ones: they "must relate to and accurately describe the factors actually considered or scored by a creditor", also a factor whose link with paying back "may not be clear to the applicant", such as "age of automobile". A reason may name just the factor; it need not say what was wrong with it: the notice "may say 'length of residence' rather than 'too short a period of residence.'"

        **Measured against whom?** Regulation B does not say how a lender with a scoring model picks its reasons, but its official comment gives two methods that will do. Both look for "the factors for which the applicant's score fell furthest below the average score for each of those factors". Method 1 takes the average of the applicants "whose total score was at or slightly above the minimum passing score", method 2 the average of "all applicants". This course reads a factor's score in a logistic regression as what it adds to the logit: its weight times its value, as in [Kinds of explanation](/course/open/rai-kinds-of-explanation?lang=en). The gap of a feature is then its weight times the applicant's value minus the average of the reference group, so `weight * (value - average)`. A gap below 0 counts against the applicant, and the most negative gap is the principal reason.

        **Why this, and not that.** Tim Miller (2019) went through what philosophy and psychology know about how people explain. Explanations are contrastive: "people do not ask why event P happened, but rather why event P happened instead of some event Q." People select: they are "adept at selecting one or two causes" from many. And causes work better than numbers: "referring to probabilities or statistical relationships in explanation is not as effective as referring to causes." Molnar's book turns this into advice: "give only 1 to 3 reasons, even if the world is more complex." Reasons measured against a reference group are contrastive in this sense: why a no for you, and not the yes of an applicant who just made it. That reading is this course's own; the law does not use the word.

        **A right to an explanation?** The GDPR gives a person a right not to be subject to a decision "based solely on automated processing" that has legal effects or affects them in a similarly significant way, with exceptions (Article 22). Where such a decision is made, the person gets "meaningful information about the logic involved" (Articles 13 to 15). The word explanation is not in those articles. It appears in recital 71, one of the numbered considerations at the start of the law, which names the "automatic refusal of an online credit application" and a right "to obtain an explanation of the decision reached". In 2017 Sandra Wachter, Brent Mittelstadt and Luciano Floridi argued that the GDPR only gives "a 'right to be informed'". Andrew Selbst and Julia Powles answered: "This is a right to explanation, whether one uses the phrase or not."

        **What the courts said.** In February 2025 the Court of Justice of the EU ruled in a case about a credit assessment by the company Dun & Bradstreet Austria (case C-203/22): Article 15(1)(h) gives "a genuine right to an explanation". Handing over "a complex mathematical formula" does not meet it. What the Court called transparent enough is telling the person "the extent to which a variation in the personal data taken into account would have led to a different result": a contrastive explanation. A company that calls its model a trade secret has to give the information to the supervisory authority or the court, which weighs the interests and decides how much the person gets. In 2023 the Amsterdam Court of Appeal ordered the company Ola to tell its drivers on which factors, weighted how, it based their fraud probability scores, their earning profiles and the rides they were given, with a penalty of 3,000 euros a day if it did not; it did not have to be the full algorithm. And Article 86 of the AI Act gives a person significantly affected by a decision based on a high-risk AI system listed in its Annex III, credit scoring among them, a right to "clear and meaningful explanations of the role of the AI system", but only where other EU law does not already give one.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **reason code** | one of the short, fixed reasons a lender sends with a no, each naming a factor that weighed against the applicant; the law says specific reasons |
        | **contrastive explanation** | an explanation of why the outcome was this rather than something else, such as why a no and not a yes |
        | **right to explanation** | the right of a person about whom an automated decision was made to be told how it was reached; in the EU it rests on the GDPR, and for decisions based on the high-risk AI systems of its Annex III also on the AI Act |
        """),
        L.watch_out("Different methods can give different reasons. For the applicant of this lesson, the two methods of Regulation B name a different principal reason: age against all applicants, education against the applicants just above the cut-off. And a reason is only as true as the method behind it. In 2022 the Consumer Financial Protection Bureau (CFPB), the US consumer finance regulator, wrote that post-hoc explanations \"approximate models\" and that lenders \"must still be able to validate the accuracy of those approximations\". The CFPB withdrew that circular on 12 May 2025; the rule it explained, specific reasons that describe the real factors, did not change."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find the reasons for a no from a logistic regression:

    1. Pick a reference group: the applicants at or just above the cut-off, or all applicants.
    2. Average each feature over that group.
    3. For each feature, work out the gap: its weight times (the applicant's value minus that average).
    4. The features with a gap below 0 are the reasons, the most negative first. Give the few that weigh most; naming the factor is enough.
    """)
    return


@app.cell(hide_code=True)
def picture_references():
    mo.vstack([
        mo.md("**Two reference groups, two principal reasons**"),
        mo.Html(C.two_references("en")),
        mo.md(r"""
        One applicant: aged 30, a high-school diploma, 40 hours of work a week. Left the gaps against the average of all 48,842 applicants: age weighs most against. Right against the average of the 1,608 applicants just above the cut-off, who are older, more educated and work more hours: now education weighs most against. Both are methods Regulation B allows.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The model of this lesson is the course's own. It uses the table UCI Adult, which [The four-fifths rule](/course/open/rai-80-percent-rule?lang=en) also uses: 48,842 people from a US survey of 1994. From three columns, `age`, `education_num` and `hours_per_week`, it predicts who earns more than 50,000 dollars a year, and we read its no as the no of a lender. `education_num` is the level of education as a number from 1 to 16: 9 is a high-school diploma, 13 a bachelor's degree. `.to_numpy()` turns the three columns into a NumPy array, one row per person. `LogisticRegression` from scikit-learn learns the way `LinearRegression` does in [The same model with scikit-learn](/course/open/ml-sklearn?lang=en); its `coef_` is a table with one row, so the weights are `coef_[0]`. Run the cell: it fits the model, rounds its weights, and works out the gaps of one applicant against all applicants, method 2. It takes a second or two.
    """)
    return


@app.cell
def example_gaps():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    # 48,842 people from a US survey of 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _names = ["age", "education_num", "hours_per_week"]
    _X = _people[_names].to_numpy()
    _y = _people["income"] == ">50K"

    # the model: a logistic regression on three columns, its weights rounded
    _model = LogisticRegression(max_iter=1000)   # more steps than the default, so it finishes learning
    _model.fit(_X, _y)
    _weights = _model.coef_[0].round(3)
    _bias = round(_model.intercept_[0], 2)
    print("weights", _weights)
    print("bias", _bias)

    # an applicant aged 30, with a high-school diploma, who works 40 hours a week
    _person = np.array([30, 9, 40])
    print("logit", round(_person @ _weights + _bias, 3))

    # method 2: the gaps against the average of all applicants
    _gaps = _weights * (_person - _X.mean(axis=0))
    print("gaps", _gaps.round(3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The weights are 0.046 per year of age, 0.345 per level of education and 0.042 per hour, with the bias −8.48. The applicant's logit is −2.315: a no. The model says yes to 5,831 of the 48,842 people; its yes or no is right for 78.9 percent of all 48,842; always saying no would be right for 76.1 percent. A weak model, but enough to show how reasons are picked.

    Against all applicants, age weighs most against this applicant (−0.398), then education (−0.372); hours hardly count (−0.018). That is method 2. Exercise 1 does method 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_gaps():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Against the applicants just above the cut-off

        The box starts with the table, the model and the applicant of the worked example. `logits` holds the logit of every applicant, and `near` is `True` for the 1,608 applicants with a logit from 0 up to 0.25: they got a yes, but only just. The `&` in that line combines two arrays of `True` and `False` element by element: `True` only where both are `True`. Work out `gaps` by method 1. First `reference`: the average of each column over the applicants in `near`. Then `gaps`: each weight times (the applicant's value minus that average). The check also tries an applicant aged 52.
        """),
        L.task("Work out `reference`, the average of each column over the applicants in `near`, and then `gaps`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gaps():
    box_gaps = L.code_box("gaps", lines=21)
    box_gaps
    return (box_gaps,)


@app.cell(hide_code=True)
def check_gaps(box_gaps):
    result_gaps = L.check("gaps", box_gaps.value)
    result_gaps
    return (result_gaps,)


@app.cell(hide_code=True)
def hints_gaps():
    L.hints(
        "gaps",
        [
            "Two steps: first the averages of the reference group, then the gaps, as in the last lines of the worked example.",
            "`X[near]` keeps the rows of the applicants in `near`, and `.mean(axis=0)` averages each column. Then multiply `weights` by `person - reference`.",
            "`reference = X[near].mean(___)`",
        ],
        why="`reference = X[near].mean(axis=0)`, then `gaps = weights * (person - reference)`. That gives age `-0.789`, education `-1.344` and hours `-0.302`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_reasons():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. From gaps to reasons

        `gaps` holds your result of exercise 1, rounded, and `names` the three features. Make the list `reasons`: the names of the features with a gap below 0, the most negative first. `np.argsort(gaps)` gives the positions of the gaps from small to large, as in [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en). The check also tries the gaps of the applicant aged 52.
        """),
        L.task("Make `reasons`: the names of the features with a gap below 0, the most negative first."),
    ])
    return


@app.cell(hide_code=True)
def exercise_reasons():
    box_reasons = L.code_box("reasons", lines=11)
    box_reasons
    return (box_reasons,)


@app.cell(hide_code=True)
def check_reasons(box_reasons):
    result_reasons = L.check("reasons", box_reasons.value)
    result_reasons
    return (result_reasons,)


@app.cell(hide_code=True)
def hints_reasons():
    L.hints(
        "reasons",
        [
            "Go over the gaps from the most negative up, and keep the name of each gap below 0.",
            "Start with `reasons = []`. Loop with `for i in np.argsort(gaps):`, and inside it, if `gaps[i] < 0`, append `names[i]`.",
            "`reasons.append(___)`",
        ],
        why="`reasons = []`, then `for i in np.argsort(gaps):` with `if gaps[i] < 0:` and inside it `reasons.append(names[i])`. That gives `['education_num', 'age', 'hours_per_week']`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_contrastive():
    mo.md(r"""
    ### Exercise 3. What people ask for
    """)
    return


@app.cell(hide_code=True)
def choice_contrastive_cell():
    choice_contrastive = L.choice(
        "contrastive",
        question="An applicant asks the bank: why did I get a no? Which answer fits best with what Miller found people want from an explanation?",
        options={
            "all": "All three weights and the bias, with your logit",
            "right": "The two features where you fell short of a yes",
            "probability": "Your probability of a yes was 0.09",
        },
        correct="right",
        why={
            "all": "Complete, but not selected: Miller found that people pick \"one or two causes\", and Molnar advises 1 to 3 reasons. And it does not say why a no rather than a yes. See *Why this, and not that* in *The idea*.",
            "right": "Short and contrastive: a few reasons, measured against the applicants who got a yes. That is what Miller found people ask for, and what your `reasons` of exercise 2 give.",
            "probability": "A probability says how sure the model is, not why. Miller: \"referring to probabilities or statistical relationships in explanation is not as effective as referring to causes.\" See *Why this, and not that* in *The idea*.",
        },
    )
    choice_contrastive
    return (choice_contrastive,)


@app.cell(hide_code=True)
def text_exercise_court():
    mo.md(r"""
    ### Exercise 4. What the court asked for
    """)
    return


@app.cell(hide_code=True)
def choice_court_cell():
    choice_court = L.choice(
        "court",
        question="In 2025 the Court of Justice of the EU ruled on a credit assessment made by automated means. Under Article 15(1)(h) of the GDPR, what does the person who was assessed get?",
        options={
            "formula": "The full formula of the model",
            "right": "How their data led to this result",
            "secret": "Nothing, if the model is a trade secret",
        },
        correct="right",
        why={
            "formula": "The Court said that handing over \"a complex mathematical formula\" does not meet the requirements: it is not a concise and intelligible explanation. See *What the courts said* in *The idea*.",
            "right": "The person gets an explanation of the procedure and principles actually applied to their data, for example \"the extent to which a variation in the personal data taken into account would have led to a different result\": a contrastive explanation.",
            "secret": "A trade secret does not end the right: the company has to give the information to the supervisory authority or the court, which weighs the interests and decides how much the person gets. See *What the courts said* in *The idea*.",
        },
    )
    choice_court
    return (choice_court,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Kinds of explanation](/course/open/rai-kinds-of-explanation?lang=en) you split a logit into what each feature adds: a local explanation. In this lesson you measured those parts against a reference group and turned the most negative into reasons, short and contrastive, as a lender sends them with a no. The next lesson, [Glass-box models](/course/open/rai-glass-box-models?lang=en), is about models whose every step a person can follow. Later in this module, [What would have to change?](/course/open/rai-counterfactuals?lang=en) asks the question from the other side: what change would have given a yes?
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_contrastive, choice_court, result_gaps, result_reasons):
    L.wrap_up(result_gaps, result_reasons, choice_contrastive, choice_court)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
