# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-glass-box-models. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-glass-box-models", "en")
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
def quick_check_black():
    quick_black = L.choice(
        "quick-black",
        question="You can give a model inputs and see its outputs, but nothing of how it gets from one to the other. What is such a model called?",
        options={
            "right": "A black box",
            "agnostic": "A model-agnostic model",
            "global": "A global explanation",
        },
        correct="right",
        why={
            "right": "A black box: you see what goes in and what comes out. Only a model-agnostic method, which needs nothing but inputs and outputs, can still explain it.",
            "agnostic": "Model-agnostic describes a method that explains, not a model: a method that needs only inputs and outputs. The model it explains here is a black box. See [Kinds of explanation](/course/open/rai-kinds-of-explanation?lang=en).",
            "global": "A global explanation describes how a model behaves overall; it is not a kind of model. A model you can only watch from the outside is a black box. See [Kinds of explanation](/course/open/rai-kinds-of-explanation?lang=en).",
        },
    )
    quick_black
    return (quick_black,)


@app.cell(hide_code=True)
def quick_check_design():
    quick_design = L.choice(
        "quick-design",
        question="Which of these explanations is read from the model itself, and not made afterwards?",
        options={
            "nearby": "A small linear model fitted around one prediction",
            "right": "The questions of a decision tree",
            "shuffle": "Shuffling one column",
        },
        correct="right",
        why={
            "nearby": "A small model fitted around one prediction is made afterwards, by asking the finished model: a post-hoc explanation. See [Kinds of explanation](/course/open/rai-kinds-of-explanation?lang=en).",
            "right": "The questions are the tree itself: you read them, nothing is made afterwards. That is interpretability by design, the subject of this lesson.",
            "shuffle": "Shuffling a column and seeing how much worse the model gets is done afterwards, by asking the finished model. See [Kinds of explanation](/course/open/rai-kinds-of-explanation?lang=en).",
        },
    )
    quick_design
    return (quick_design,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    COMPAS is a commercial score that predicts whether someone will be arrested again; you may know it from [COMPAS and a hiring tool](/course/open/rai-case-compas-amazon?lang=en). How it works is not public: Cynthia Rudin (2019) calls a model like that a black box because it is "proprietary". Rudin set beside it a model anyone can read: three rules, on age and the number of past crimes, one of them also on sex. The question: if three rules could do the job, why trust a secret model? A model whose every step a person can follow is called a glass box: there is nothing to explain afterwards, because you can read it.

    **An everyday example.** A tax form leads you line by line to what you owe: you can check every step, and if a line is wrong, you can see which one. An experienced accountant may come up with the same number from experience, but you cannot follow how.
    """)
    return


@app.cell(hide_code=True)
def predict_three():
    guess_three = L.guess(
        "three",
        question="Rudin compared COMPAS with the three rules on data from Broward County, Florida. How well did the three rules predict who would be arrested again?",
        options={
            "much": "Much worse than COMPAS",
            "little": "A little worse",
            "same": "About as well",
        },
        reveal="**About as well.** Rudin: \"The COMPAS model is equally accurate for recidivism prediction as the very simple three rule interpretable machine learning model involving only age and number of past crimes\". \"Both models have similar true and false positive rates and true and false negative rates on data from Broward County, Florida.\" Why a simple model can do as well as a complicated one is the subject of [Many good models](/course/open/rai-rashomon?lang=en), later in this module.",
    )
    guess_three
    return (guess_three,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A glass box.** InterpretML, a library by Harsha Nori and colleagues (2019), uses the word. It sorts methods into "glassbox" models, "machine learning models designed for interpretability (ex: linear models, rule lists, generalized additive models)", and blackbox techniques that explain existing systems. Its documentation: "Glassbox models are structured for direct interpretability, meaning the explanations that are generated are exact and human interpretable. This is in contrast to blackbox models, where explanations are generally approximate." Other sources say interpretable model, transparent model or white box; Molnar's book prefers interpretable model. This course says glass-box model.

        **Following it by hand.** Zachary Lipton (2016) gives two tests. The first: "a human should be able to take the input data together with the parameters of the model and in reasonable time step through every calculation required to produce a prediction." The second: "each part of the model - each input, parameter, and calculation - admits an intuitive explanation", which "requires that inputs themselves be individually interpretable". A weight for a column called `f3` passes the first test and fails the second.

        **Reading a linear model.** Molnar: "An increase of feature x_j by one unit increases the prediction for y by β_j units when all other feature values remain fixed." (Molnar writes β_j for the weight.) A logistic regression adds up to a logit instead, and the logit is the log of the odds, as in [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en). So one unit more multiplies the odds by $e^{w}$: "the estimated odds change by a factor of exp(β_j)." The weights add up; their effect on the odds multiplies, and on the probability it is neither: "the interpretation of the weights is multiplicative and not additive."

        **Reading a tree.** Each path from the top to a leaf is a rule. Molnar: "If feature x_j is [smaller/bigger] than threshold c AND … then the predicted outcome is the mean value of y of the instances in that node." For a tree that says yes or no, that mean is the share of yes in the leaf, and the leaf says yes when the share is above one half. A tree of depth $d$ has at most $2^d$ leaves, so "Decision trees are very interpretable – as long as they are short." Molnar also warns: "Trees are also quite unstable." A slightly different table can grow a different tree.

        **How small is small enough?** Rudin: "humans can handle at most 7±2 cognitive entities at once". Rudin cites George Miller's "The magical number seven, plus or minus two" (1956) and Nelson Cowan's "The magical mystery four" (2010); as Cowan's title suggests, newer work puts the number lower, at about four. Lipton adds that no kind of model is a glass box by itself: "neither linear models, rule-based systems, nor decision trees are intrinsically interpretable." Deep trees, long rule lists and models with thousands of features "could all be considered less transparent than comparatively compact neural networks."

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **glass-box model** | a model that is interpretable by design, such as a short decision tree or a linear model with few features, so a person can follow every step from input to output |
        | $e^{w}$ | the factor by which one unit more of a feature multiplies the odds, in a logistic regression with weight $w$ |
        """),
        L.watch_out("A weight is not the effect of its feature on its own. It holds with all other features fixed, and it changes with which other features are in the model. Lipton's example: \"associations between flu risk and vaccination might be positive or negative depending on whether the feature set includes indicators of old age, infancy, or immunodeficiency.\" And a glass box can learn something wrong just as well; you can only see it more easily, a story the next lesson tells."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To check whether a person can follow a model:

    1. Count its parts: weights, questions, leaves. A handful can be followed; thousands cannot.
    2. Check that each input means something on its own, with a name and a unit.
    3. Step through one prediction. A linear model: multiply and add. A logistic regression: also read $e^{w}$ as the factor on the odds. A tree: follow the questions to a leaf.
    4. Read each weight with its unit, and with the other features fixed.
    """)
    return


@app.cell(hide_code=True)
def picture_tree():
    mo.vstack([
        mo.md("**A glass box, drawn whole**"),
        mo.Html(C.whole_tree("en")),
        mo.md(r"""
        The tree of the worked example, on the 48,842 people of UCI Adult. Two questions decide every prediction. Only one leaf says yes: education above 12.5 and older than 29.5, where 56.5 percent earn more than 50,000 dollars. On the left the tree asks about age although both leaves say no: the question makes the leaves purer, 4 percent against 22.6 percent, without changing what they say.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Two glass boxes on the three columns of [Reasons for a decision](/course/open/rai-reason-codes?lang=en): the logistic regression of that lesson, and a decision tree of depth 2 as in [Question after question: the decision tree](/course/open/ml-decision-tree?lang=en). `.to_numpy()` turns the three columns into a NumPy array, one row per person. Run the cell: it prints the weights and their factors on the odds, the tree as `export_text` writes it, and how often each model is right. It takes a second or two.
    """)
    return


@app.cell
def example_glass():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, export_text

    # 48,842 people from a US survey of 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _names = ["age", "education_num", "hours_per_week"]
    _X = _people[_names].to_numpy()
    _y = _people["income"] == ">50K"

    # glass box 1: a logistic regression, one weight per column
    _model = LogisticRegression(max_iter=1000)
    _model.fit(_X, _y)
    _weights = _model.coef_[0].round(3)
    print("weights", _weights)
    print("odds factors", np.exp(_weights).round(3))
    print("logistic regression right", round(np.mean(_model.predict(_X) == _y), 3))

    # glass box 2: a decision tree with at most two questions on any path
    _tree = DecisionTreeClassifier(max_depth=2, random_state=0)
    _tree.fit(_X, _y)
    print(export_text(_tree, feature_names=_names))
    print("tree right", round(np.mean(_tree.predict(_X) == _y), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    One more level of education multiplies the odds of earning more than 50,000 dollars by 1.412, one more year of age by 1.047, one more hour a week by 1.043, each with the other two fixed. The logistic regression is right for 78.9 percent of the people, the tree for 78.7 percent; always saying no would be right for 76.1 percent. Neither is a strong model, but you can read both completely: three weights and a bias, or three questions and four leaves. That is what makes them glass boxes, not how well they predict.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_follow():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Follow the tree by hand

        The box starts with the tree of the worked example, as comments. Write the function `says_yes(age, education_num, hours_per_week)`: it follows the tree with `if` and `else`, as you would by hand, and returns `True` or `False`, as the leaf says. Each `|` further to the right is one question deeper; a line with `<=` leads to the part where the question holds. The check calls your function for the applicant of [Reasons for a decision](/course/open/rai-reason-codes?lang=en), aged 30 with education level 9 and 40 hours, and for twelve other people; the picture tries it on everyone in the table.
        """),
        L.task("Write `says_yes(age, education_num, hours_per_week)`: follow the questions of the tree and return what its leaf says."),
    ])
    return


@app.cell(hide_code=True)
def exercise_follow():
    box_follow = L.code_box("follow", lines=23)
    box_follow
    return (box_follow,)


@app.cell(hide_code=True)
def check_follow(box_follow):
    result_follow = L.check("follow", box_follow.value)
    result_follow
    return (result_follow,)


@app.cell(hide_code=True)
def hints_follow():
    L.hints(
        "follow",
        [
            "Start at the top line of the tree: one question with two parts. In each part comes one more question, and then a leaf.",
            "Write `if education_num <= 12.5:`, and inside it a second `if` about `age` with `return False` in both parts. Under `else:` comes the other question about `age`, with its own threshold.",
            "`if age <= 29.5:` with `return False`, and under `else:`, `return ___`",
        ],
        why="`if education_num <= 12.5:` with inside it `if age <= 30.5: return False` and `else: return False`; then `else:` with `if age <= 29.5: return False` and `else: return True`, each `return` on its own line. In short: `return education_num > 12.5 and age > 29.5`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_odds():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Four levels of education

        `weights` holds the weights of the logistic regression: age, education, hours. Two people are the same age and work the same hours; one has a high-school diploma (education level 9), the other a bachelor's degree (level 13). Work out `factor`: how many times larger the odds of a yes are for the second. The check also tries other weights.
        """),
        L.task("Work out `factor`: how many times the odds of a yes grow from education level 9 to 13."),
    ])
    return


@app.cell(hide_code=True)
def exercise_odds():
    box_odds = L.code_box("odds", lines=7)
    box_odds
    return (box_odds,)


@app.cell(hide_code=True)
def check_odds(box_odds):
    result_odds = L.check("odds", box_odds.value)
    result_odds
    return (result_odds,)


@app.cell(hide_code=True)
def hints_odds():
    L.hints(
        "odds",
        [
            "One level more multiplies the odds by $e$ to the power of the weight. Four levels more multiply four times.",
            "The weight of education is `weights[1]`. Four levels add four times that weight to the logit; the odds change by `np.exp` of what is added.",
            "`factor = np.exp(weights[1] * ___)`",
        ],
        why="`factor = np.exp(weights[1] * 4)`, or `np.exp(weights[1]) ** 4`: about `3.975`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_glass():
    mo.md(r"""
    ### Exercise 3. Which can a person follow?

    Put each model in its group: can a person follow a prediction step by step, as it stands?
    """)
    return


@app.cell(hide_code=True)
def sort_glass_cell():
    sort_glass = L.sort("glass")
    sort_glass
    return (sort_glass,)


@app.cell(hide_code=True)
def check_glass(sort_glass):
    result_glass = L.check("glass", sort_glass)
    result_glass
    return (result_glass,)


@app.cell(hide_code=True)
def hints_glass():
    L.hints(
        "glass",
        [
            "Two questions for each: how many parts does it have, and does each part mean something on its own?",
            "Three models are small and have parts with a meaning. A tree of depth 20 can have 2 to the power 20 leaves, more than a million.",
            "A person can follow: the short tree, the points and \\_\\_\\_.",
        ],
        why="**A person can follow it:** the short tree, the three named weights, the points. **Not as it stands:** the deep tree, the 5,000 weights, the four made-up columns, the small network.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_scale():
    mo.md(r"""
    ### Exercise 4. Comparing weights
    """)
    return


@app.cell(hide_code=True)
def choice_scale_cell():
    choice_scale = L.choice(
        "scale",
        question="In the logistic regression, education has the weight 0.345 and age 0.046. Does education count 7.5 times as much as age?",
        options={
            "times": "Yes, its weight is 7.5 times as large as age's",
            "right": "No, a level and a year are different units",
            "less": "No, age counts more than education",
        },
        correct="right",
        why={
            "times": "Each weight is per unit of its own feature: one level of education, one year of age. Molnar: \"Don't compare weights of features with different scales\". Measured in months, age would get a weight 12 times smaller, without the model changing at all. See *Reading a linear model* in *The idea*.",
            "right": "A weight is per unit: per level of education, per year of age. Measured in months, age would get a weight 12 times smaller, and nothing about the model would have changed. Molnar: \"Don't compare weights of features with different scales\".",
            "less": "Nothing in the weights says that either. Each weight is per unit of its own feature, a level of education or a year of age, and those units cannot be compared. See *Reading a linear model* in *The idea*.",
        },
    )
    choice_scale
    return (choice_scale,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Kinds of explanation](/course/open/rai-kinds-of-explanation?lang=en) you read an explanation from the model itself; a glass box is a model where that explanation is the whole model. In [Reasons for a decision](/course/open/rai-reason-codes?lang=en) you used its weights to give reasons. The next lesson, [Additive models](/course/open/rai-additive-models?lang=en), keeps a glass box but lets each feature bend: a curve per feature instead of a straight line. It also tells the story of a glass box that learned something dangerous, and of how its makers could see it. [Many good models](/course/open/rai-rashomon?lang=en) asks why a simple model can often predict as well as a complicated one.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_scale, result_follow, result_glass, result_odds):
    L.wrap_up(result_follow, result_odds, result_glass, choice_scale)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
