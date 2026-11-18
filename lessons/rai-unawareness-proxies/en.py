# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-unawareness-proxies. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-unawareness-proxies", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [The four-fifths rule](/course/open/rai-80-percent-rule?lang=en) and [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_ratio():
    quick_ratio = L.choice(
        "quick-ratio",
        question="A model gives a yes to 7 percent of the women and 25 percent of the men. What does the four-fifths rule say?",
        options={
            "right": "A signal: 0.28 is below 0.8",
            "points": "Nothing yet: the gap is only 18 percentage points",
            "column": "A signal only if the model sees the sex column",
        },
        correct="right",
        why={
            "right": "7 divided by 25 is 0.28, far below four-fifths. The rule looks at who gets a yes, not at how the model got there.",
            "points": "The rule divides the rates, it does not subtract them: 7 / 25 = 0.28. See *The idea* in [The four-fifths rule](/course/open/rai-80-percent-rule?lang=en).",
            "column": "The rule only looks at who gets a yes, not at the columns the model used. See *The idea* in [The four-fifths rule](/course/open/rai-80-percent-rule?lang=en).",
        },
    )
    quick_ratio
    return (quick_ratio,)


@app.cell(hide_code=True)
def quick_check_class():
    quick_class = L.choice(
        "quick-class",
        question="A logistic regression gives one student a probability of 0.73 of passing. Which class does it predict for that student?",
        options={
            "fail": "A fail, as 0.73 is below 1",
            "right": "A pass, as 0.73 is above 0.5",
            "none": "None: it only gives a probability",
        },
        correct="right",
        why={
            "fail": "The model predicts class 1 wherever the probability is above 0.5, not only at 1. See *The idea* in [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en).",
            "right": "Above 0.5 the logit is above 0: the student is on the pass side of the decision boundary.",
            "none": "Logistic regression gives a probability, and predicts class 1 where it is above 0.5. See *The idea* in [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en).",
        },
    )
    quick_class
    return (quick_class,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In 2014 Amazon began to build a model that rated job applicants from one to five stars. It learned from the resumes sent to the company over ten years, most of them from men. By 2015 the company saw that it did not rate applicants in a gender-neutral way: it marked down resumes with the word *women's*, as in *women's chess club captain*. The team made the model ignore such words, but could not be sure it would find no other way to sort the applicants that could discriminate, and the team was disbanded. The lesson [COMPAS and a hiring tool](/course/open/rai-case-compas-amazon?lang=en) tells the whole story. A word on a resume was enough to stand in for sex. This lesson shows that leaving out the column for sex does not stop a model from treating women and men differently.

    **An everyday example.** A job ad says it does not ask your age. But the form asks for the year you got your diploma. Whoever reads the forms can still work out roughly how old you are.
    """)
    return


@app.cell(hide_code=True)
def predict_unaware():
    guess_unaware = L.guess(
        "unaware",
        question="In the table UCI Adult of [The four-fifths rule](/course/open/rai-80-percent-rule?lang=en), a model that predicts who earns more than 50,000 dollars gives a yes to 7 percent of the women and 25 percent of the men. It is trained again without the column `sex`. What do you think the women get then?",
        options={
            "men": "About 25 percent, as many as the men",
            "right": "About 7 percent, nearly no change",
            "half": "About 16 percent",
        },
        reveal="**Nearly no change: still 7 percent.** The other columns tell the model a great deal about who is a woman, so leaving out `sex` hardly matters. The demo below shows it, and how well the other columns give away sex.",
    )
    guess_unaware
    return (guess_unaware,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Leaving the column out.** The first idea of many teams: give the model no column for sex or ethnicity, and it cannot use them. Matt Kusner and colleagues (2017) call this fairness through unawareness: "An algorithm is fair so long as any protected attributes A are not explicitly used in the decision-making process." Here A is a column such as sex. The flaw, in their words: the other features "can contain discriminatory information analogous to A that may not be obvious at first."

    **Proxies.** The EU Agency for Fundamental Rights (FRA) wrote in 2022: "A proxy is a seemingly neutral piece of information that is nevertheless strongly related to a protected characteristic. For example, shoe size as a proxy for gender or names as a proxy for ethnicity." There is "a potentially limitless number of proxies", so discrimination through them is harder to prevent. The *Handreiking non-discriminatie by design* (2021), a guide written for the Dutch Ministry of the Interior, gives the postcode as a proxy for ethnicity, and a rule that lets through only applicants taller than 1.70 meters because everyone on the current staff is: women are on average shorter, so fewer of them get through.

    **Many small hints.** A proxy need not be one obvious column. The book *Fairness and Machine Learning* notes that visiting pinterest.com had a small statistical correlation with being female in the United States: too small on its own to tell someone's sex. But "several features that are slightly predictive of the sensitive attribute can be used to build high accuracy classifiers for that attribute." So "if a classifier trained on the original data uses the sensitive attribute and we remove the attribute, the classifier will then find a redundant encoding in terms of the other features."

    **Redlining.** Between 1935 and 1940 the US Home Owners' Loan Corporation drew color-coded maps of 239 cities, grading neighborhoods from A, "Best", in green, to D, "Hazardous", in red. Research that the Federal Reserve Bank of St. Louis cites found the largest concentrations of African Americans in the lowest-graded areas. Refusing or limiting loans to whole neighborhoods is called redlining, after that red; Cynthia Dwork and colleagues (2012) call it "a well-known form of discrimination based on redundant encoding." It is illegal in the US today. In 2021 Trustmark National Bank settled a claim of the Justice Department that it had redlined Black and Hispanic neighborhoods of Memphis, with a fund of 3.85 million dollars to subsidize loans and a penalty of 5 million.

    **Try it.** The cell below uses the table UCI Adult of [The four-fifths rule](/course/open/rai-80-percent-rule?lang=en): 48,842 people from a US survey of 1994. It trains a logistic regression on the training set to predict who earns more than 50,000 dollars, three times: with the column `sex`, without it, and also without `relationship` and `marital_status`. `LogisticRegression` from scikit-learn works like `LinearRegression` in [The same model with scikit-learn](/course/open/ml-sklearn?lang=en): `fit`, then `predict`, which gives `True` where the probability is above 0.5. `pd.get_dummies` turns each column of words into one-hot columns, one per value, as in [One-hot vectors for classes](/course/open/ml-one-hot?lang=en). For each model the cell prints the yes rates of women and men in the test set, and how often the model is right there. Then it trains a model that guesses from the other columns who is a man. Run the cell; it takes a few seconds.
    """)
    return


@app.cell
def demo_models():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    # 48,842 people from a US survey of 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _train = _people["split"] == "train"
    _test = _people["split"] == "test"
    _tested = _people[_test]
    _words = ["workclass", "marital_status", "occupation", "relationship", "race"]


    def _features(words):
        # each column of words becomes one-hot columns; then every column runs from 0 to 1
        columns = ["age", "education_num", "hours_per_week"] + words
        X = pd.get_dummies(_people[columns], columns=words, dtype=float)
        return X / X.max()


    def _model_says(name, words):
        X = _features(words)
        rich = _people["income"] == ">50K"
        model = LogisticRegression(max_iter=1000)   # more steps than the default, so it finishes learning
        model.fit(X[_train], rich[_train])
        yes = model.predict(X[_test])
        picked = _tested[yes]
        rates = picked.groupby("sex").size() / _tested.groupby("sex").size()
        right = (yes == rich[_test]).mean()
        print(name, "women", round(rates["Female"], 2), "men", round(rates["Male"], 2),
              "right", round(right, 2))


    _model_says("with sex:   ", _words + ["sex"])
    _model_says("without sex:", _words)
    _model_says("also without relationship and marital_status:", ["workclass", "occupation", "race"])

    # can the other columns tell who is a man?
    _man = _people["sex"] == "Male"
    _X = _features(_words)
    _guesser = LogisticRegression(max_iter=1000)
    _guesser.fit(_X[_train], _man[_train])
    _guessed = _guesser.predict(_X[_test])
    print("guessing sex, right:", round((_guessed == _man[_test]).mean(), 2))
    print("always 'man', right:", round(_man[_test].mean(), 2))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.picture({"title": "Yes rates of women and men",
                           "rows": ["In the data: the labels", "Model with the column sex", "Model without the column sex",
                                    "Also without relationship and marital status"],
                           "xlabel": "yes rate", "key": "open: women, filled: men", "decimal": ".",
                           "alt": "Four pairs of bars of the yes rates of women and men in the test set. In the data: 0.11 and 0.30. The model with the column sex: 0.07 and 0.25. Without it: 0.07 and 0.25. Also without relationship and marital status: 0.08 and 0.17."})),
        mo.md(r"""
        With the column `sex`, the model gives a yes to 0.07 of the women and 0.25 of the men; without it, to the same 0.07 and 0.25, and it is right about as often. The other columns know who is a woman: a model that guesses sex from them is right 0.84 of the time, where always guessing "man" is right 0.67. The clearest is `relationship`: each person's place in the household, such as Husband or Wife. In Exercise 1 you measure how much it gives away.

        Leaving out `relationship` and `marital_status` as well narrows the gap to 0.08 against 0.17, but the model is right less often, 0.8, and women still get a yes half as often as men. Part of the gap is in the labels themselves: in the data, 0.11 of the women and 0.30 of the men earn more than 50,000 dollars. Toon Calders and Sicco Verwer (2010) found on this same table that another kind of model, trained without the column for sex, even favored men more than the data did: a yes for 38 percent of the men and 10 percent of the women, where the data had 30 and 11. *Fairness and Machine Learning* puts it in general terms: leaving out such columns "typically has a negligible impact on a classifier when rich datasets are available." Chasing proxies one by one can even backfire: when some US laws banned questions about criminal history in hiring, employers increased racial discrimination, as Agan and Starr found.

        Dwork and colleagues called redundant encoding "a successful attack against 'fairness through blindness'", and named their own paper *Fairness Through Awareness*: their method decides with knowledge of what makes two people alike for the task. The cell above shows a plainer kind of awareness: to see any of this, it needed the column `sex`. It left it out of the model, but kept it to check.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **fairness through unawareness** | leaving a column such as sex or ethnicity out of the features of a model, hoping it then treats the groups alike; on its own it does not work |
        | **proxy** | a feature that seems neutral but carries information about such a column, such as a postcode about ethnicity |
        | **redundant encoding** | when the other features together give a left-out column away |
        | **redlining** | refusing or limiting loans to whole neighborhoods, which hit people of color: the neighborhood stood in for race |
        """),
        L.watch_out("Leaving a column out of the features is not the same as throwing it away. Without it, nobody can count the yes rates per group, as the cell above did, so nobody can see that the model still treats the groups differently."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To check a model that leaves out a column such as sex:

    1. Keep the column aside: out of the features, but in the data you test the model with.
    2. Count the yes rates per group, and divide them as in the four-fifths rule.
    3. Look for proxies: for each value of a column, the share of each group. A value with a share near 0 or 1 gives the group away.
    4. Let a model guess the column from the other features. If it does much better than always guessing the largest group, the column is still in the data.
    5. Do not count on deleting proxies one by one. The coming lessons measure fairness and change models on purpose.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The table UCI Adult again. `_column` is the column `relationship`: each person's place in the household, such as Husband, Wife or Own-child. The cell picks the people who earn more than 50,000 dollars, and divides how many of them have each value by how many people have that value. Run the cell.
    """)
    return


@app.cell
def example_rich(pd):
    # 48,842 people from a US survey of 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _column = "relationship"

    _rich = _people[_people["income"] == ">50K"]
    _share = _rich.groupby(_column).size() / _people.groupby(_column).size()
    print(_share.round(2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Husbands and wives earn more than 50,000 dollars about equally often, 0.45 and 0.47; people who live in the household as a child hardly ever, 0.01. So the column helps the model predict income. But who are the husbands and the wives? Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_proxy():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A proxy for sex

        `column` is the column `relationship`. Replace the `...` with the two lines of the worked example, with the names without their underscore: `men` and `share`. Put in `men` the rows where `sex` is `"Male"`, and in `share` the men per value of `column` divided by all the people per value. The last line is `share`, so the check sees it. The check also tries another column.
        """),
        L.task("Work out `share`: for each value of `relationship`, the share of men."),
    ])
    return


@app.cell(hide_code=True)
def exercise_proxy():
    box_proxy = L.code_box("proxy", lines=10)
    box_proxy
    return (box_proxy,)


@app.cell(hide_code=True)
def check_proxy(box_proxy):
    result_proxy = L.check("proxy", box_proxy.value)
    result_proxy
    return (result_proxy,)


@app.cell(hide_code=True)
def hints_proxy():
    L.hints(
        "proxy",
        [
            "Use the two lines of the worked example. Only the rows you pick change: not the people who earn more than 50,000 dollars, but the men.",
            "The men are the rows where the column `sex` is `\"Male\"`: `people[people[\"sex\"] == \"Male\"]`.",
            "`share = men.groupby(column).size() / ___`.",
        ],
        why="`men = people[people[\"sex\"] == \"Male\"]`, then `share = men.groupby(column).size() / people.groupby(column).size()`. That gives about 1.0 for Husband and about 0.0 for Wife.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_proxies():
    mo.md(r"""
    ### Exercise 2. What can stand in for what?

    Six pieces of data a hiring model might get. None of them names ethnicity or sex. Sort each by what it can stand in for.
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
            "Ask of each piece of data: which group has this more often, or less often?",
            "A postcode tells where someone lives, and neighborhoods are often not mixed. Which other piece of data says something about where a family comes from?",
            "Can stand in for ethnicity: postcode and \\_\\_\\_. Can stand in for sex: height, \\_\\_\\_, the gap and shoe size.",
        ],
        why="**Can stand in for ethnicity:** postcode and surname. **Can stand in for sex:** height, the club, the gap in the resume and shoe size.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_check():
    mo.md(r"""
    ### Exercise 3. Keep it to check
    """)
    return


@app.cell(hide_code=True)
def choice_check_cell():
    choice_check = L.choice(
        "check",
        question="A team drops the column `sex` from its data altogether, also from the data it tests the model with. Can it still show that its model gives women and men a yes about equally often?",
        options={
            "never": "Yes, the model never saw sex",
            "right": "No, it cannot count yes rates per sex",
            "accuracy": "Yes, if the model is right as often as before",
        },
        correct="right",
        why={
            "never": "The model without `sex` still gave women and men very different yes rates. See *Try it* in *The idea*.",
            "right": "A yes rate per group needs to know who is in which group. Leave the column out of the features, but keep it to check.",
            "accuracy": "Without `sex` the model was right about as often, and still gave men a yes more than three times as often. See *Try it* in *The idea*.",
        },
    )
    choice_check
    return (choice_check,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [The four-fifths rule](/course/open/rai-80-percent-rule?lang=en) you counted yes rates per group; here you saw that a model can give groups very different rates without ever seeing the group. In [Feedback loops](/course/open/rai-feedback-loops?lang=en) a model shaped its own data; here the data shapes the model through columns that stand in for sex. Your share per value is a first test for a proxy, and it only works because the column `sex` was kept. The next lesson, *Equal yes rates*, makes equal yes rates per group a fairness criterion of its own, and shows what it misses.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_check, result_proxies, result_proxy):
    L.wrap_up(result_proxy, result_proxies, choice_check)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
