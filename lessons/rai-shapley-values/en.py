# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-shapley-values. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-shapley-values", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [LIME: a simple model around one prediction](/course/open/rai-lime?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_weight():
    quick_weight = L.choice(
        "quick-weight",
        question="In the last lesson, LIME gave education_num a weight of 0.144 for the person aged 41. What is that weight?",
        options={
            "share": "The part of the chance that education gives",
            "right": "The slope of a line near the person",
            "count": "How often the forest asks about education",
        },
        correct="right",
        why={
            "right": "Right: a LIME weight is the slope of a straight line fitted near the person, per standard deviation of the column. It does not share the chance out over the features. The Shapley values of this lesson do.",
            "share": "That is what this lesson's Shapley values give. A LIME weight is the slope of a straight line near the person: how much the line's chance moves for one standard deviation more. See [LIME: a simple model around one prediction](/course/open/rai-lime?lang=en).",
            "count": "LIME never looks inside the forest: it only asks for its answers on made-up rows, and its weights are the slopes of a straight line through them. See [LIME: a simple model around one prediction](/course/open/rai-lime?lang=en).",
        },
    )
    quick_weight
    return (quick_weight,)


@app.cell(hide_code=True)
def quick_check_small():
    quick_small = L.choice(
        "quick-small",
        question="With kernel width 0.1, the weight of education_num was 0.081, 0.199 and 0.242 in three runs. Why did it jump?",
        options={
            "trained": "The forest was trained again each run",
            "moved": "The random rows moved the person each time",
            "right": "Together the 5000 rows counted as about 2",
        },
        correct="right",
        why={
            "right": "Right: with width 0.1 the 5000 rows count as much as only about 2 rows at the person, so each run's line rests on a handful of rows. The Shapley values of this lesson are worked out exactly for four columns: for the same background they come out the same every time.",
            "trained": "The forest is grown once, with `random_state=0`; only LIME's made-up rows change between runs. With width 0.1 they count as much as only about 2 rows at the person. See [LIME: a simple model around one prediction](/course/open/rai-lime?lang=en).",
            "moved": "The person stays where they are; the made-up rows change between runs, and with width 0.1 they count as much as only about 2 rows at the person. See [LIME: a simple model around one prediction](/course/open/rai-lime?lang=en).",
        },
    )
    quick_small
    return (quick_small,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The supervisor has LIME's answer for the person aged 41 with a bachelor's degree who got a no: with lime's own kernel width education_num weighs most, with a smaller width hours_per_week. The person asks something else: how much of the gap between their chance and other people's comes from each feature? With a fourth column, capital_gain, the forest gives this person a chance of 0.451, and 200 people of the training half get 0.225 on average. A Shapley value shares that gap of 0.226 out over the four features so that the four parts add up to it exactly, by a rule from game theory that Lloyd Shapley wrote down in 1952. With four features you can work it out exactly. But "other people" is a choice: compare with a different group, and the shares change.

    **An everyday example.** Two friends enter a pub quiz as a team and score 10 points. Alone, Ash would have scored 2 and Bo 4. How do you share the 10? Let them join in each order, and give each what the team gains when they arrive. If Ash comes first, Ash adds 2 and Bo then adds 8; if Bo comes first, Bo adds 4 and Ash then adds 6. Averaged over the two orders, Ash gets (2 + 6) / 2 = 4 and Bo (8 + 4) / 2 = 6: together 10.
    """)
    return


@app.cell(hide_code=True)
def predict_largest():
    guess_largest = L.guess(
        "largest",
        question="The four features share the gap of 0.226 between this person's chance and the background's average. Which one gets the largest share?",
        options={
            "age": "age",
            "education": "education_num",
            "hours": "hours_per_week",
            "gain": "capital_gain",
        },
        reveal="**education_num, with +0.181.** Then age, with +0.087. hours_per_week gets −0.010 and capital_gain −0.032, below zero: the person has no capital gain, and putting that 0 into every background row lowers the forest's average from 0.225 to 0.190; 22 of the 200 people have one. Together 0.226, exactly the gap. LIME also put education_num first at lime's own width, but its weights are slopes; these are shares of a chance.",
    )
    guess_largest
    return (guess_largest,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A share for each player.** In Lloyd Shapley's paper for the RAND Corporation of 1952, a game gives a value to every coalition, every group of players, with 0 for the empty one, and the question is how to share the value of all players together. The answer comes as a story: "Starting with a single member, the coalition adds one player at a time until everyone has been admitted." The order is "determined by chance, with all arrangements equally probable", and each player gets what the coalition gains when that player joins. Shapley asked for three rules: symmetry, "that the value is essentially a property of the abstract game"; efficiency, that the value "represents a distribution of the full yield of the game"; and a law of aggregation, that the value of two games added together is the sum of their values. "It is remarkable that no further conditions are required to determine the value uniquely." A player who adds nothing to any coalition then gets nothing.

        **The formula.** Erik Štrumbelj and Igor Kononenko write it as a sum over the coalitions $S$ that do not hold player $i$, with $n$ players and $s$ players in $S$:

        $$\phi_i = \sum_{S\,:\,i \notin S} \frac{s!\,(n-s-1)!}{n!}\,\bigl(v(S \cup \{i\}) - v(S)\bigr)$$

        The weight is the share of the $n!$ orders in which exactly the players of $S$ come before $i$. With four players there are 24 orders. In 6 of them $i$ comes first, so the empty coalition counts $6/24 = 1/4$; one given player is the only one before $i$ in 2 orders, $1/12$; two given players are the only ones before $i$ in 2 orders too, $1/12$; all three others in 6, $1/4$. Each of the four players has 8 coalitions without it, and all four values together need the 16 coalitions.

        **From players to features.** Štrumbelj and Kononenko (2010) made the features of a model the players: "We present a general method for explaining individual predictions of classification models." Their first example: "The prior probability of a Titanic passenger's survival is 32% and the model predicts a 67% chance of survival." A coalition's value is then how much the expected prediction changes when only the features in it are known. They took the unknown features as spread evenly over all their possible values: "we assume uniform distribution of feature values." This lesson does what Christoph Molnar describes instead: fill in each unknown feature from rows of the data, here 200 people of the training half, the background, and average. Here that average itself is a coalition's value, so the empty coalition is worth 0.225 rather than 0; taking the same number off every coalition changes none of the differences, so the Shapley values stay the same. Then the values add up to the prediction minus the average prediction over that same background. Molnar's example is an apartment: the model predicts €300,000, and "The average prediction for all apartments is €310,000." The four contributions "add up to -€10,000, the final prediction minus the average predicted apartment price."

        **SHAP.** Scott Lundberg and Su-In Lee (2017) put LIME and Shapley values in one family: an explanation as a sum, a base value plus one part for each feature. They asked for three properties: local accuracy, the base value and the parts add up to the prediction; missingness, a feature that is not there gets nothing; and consistency. "Only one possible explanation model g follows Definition 1 and satisfies Properties 1, 2, and 3": the Shapley values. They called them SHAP values, "the Shapley values of a conditional expectation function of the original model", and to compute them they allowed "feature independence" as an optional assumption, which gives the kind this lesson computes. With LIME's own choices, they showed, LIME's line "does not recover the Shapley values"; with another kernel, fitted on which features are known (0 or 1), it does. The next lesson shows the tools.

        **Which "unknown"?** Filling in from background rows is one choice of game, and there are others. Elizabeth Kumar, Suresh Venkatasubramanian, Carlos Scheidegger and Sorelle Friedler call the choice "a catch-22". Their "value function" is the game itself, the $v$ that gives each coalition its value, not the value function of reinforcement learning. Fill in from other people as they are, as here, and the model is asked about rows that may not exist: "Methods which use an interventional value function fundamentally rely on evaluating a model on out-of-distribution samples". Fill in only values that fit the known ones (the conditional kind), and "the Shapley value of the conditional value function may attribute influence to features with no interventional effect": a feature the model never uses can get a share. Mukund Sundararajan and Amir Najmi found that the different ways "give very different results, rendering the uniqueness result meaningless". Shapley's rules make the answer unique for one game; which game is your choice.

        **What a Shapley value does not say.** Molnar: "The Shapley value is NOT the difference in prediction when we remove the feature from the model." It gives no direction either: "a positive Shapley value doesn't mean that increasing the feature would increase the prediction." Kumar et al. give an example: a model $f(x) = 2 - (x - 1)^2$, with $x$ drawn around 0 with spread 1. A person with $x = 1$ gets a Shapley value of 2, the most anyone can get, which "might make this person think that increasing their x value even more will increase their score – but it will not." And a share is always against one background: "Shapley value interpretation is always in reference to the dataset that was used for replacing values for absent players." When a model multiplies independent features centered on 0, every feature gets the same share, whatever its own value (Kumar et al.). And SHAP can be fooled as LIME was, though less often: with one meaningless column, Slack et al.'s attack worked for "84% of data points in COMPAS".

        **What it costs.** Molnar: "The computation time increases exponentially with the number of features." Four features need 16 coalitions; twenty need over a million. Štrumbelj and Kononenko already took a sample of orders for that reason: "We overcome the method's initial exponential time complexity with a sampling-based approximation."

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **Shapley value** | a feature's share of the gap between one prediction and the average prediction: what it adds to a coalition, averaged over every order in which the features can join |
        | **coalition** | the features that are known; with four features there are 16, from none to all four |
        | **background** | the rows that fill in the features that are not known; the Shapley values add up to the prediction minus the model's average over them |
        | **SHAP** | Lundberg and Lee's name for the Shapley values of a model's prediction, and the library that computes them |
        | $\phi_i$ | the Shapley value of feature $i$ (phi) |
        | $s!$ | $1 \times 2 \times \dots \times s$, with $0! = 1$; in Python `math.factorial(s)` |
        """),
        L.watch_out("A Shapley value is a share of the gap between this prediction and the model's average over the background you chose. It is not what the prediction does without the feature, it does not say which way to change the feature, and with another background it changes. Say which background you used, and read the value as a share, not as a cause."),
    ])
    return


@app.cell(hide_code=True)
def explore_background_cell():
    explore_background = L.explore("background-top", "highest education_num in the background",
                                   lambda v: C.background_picture(v, "en"), start=16, values=C.TOPS, role="hyper",
                                   task="Move the slider from 16 down to 8. What happens to the share of education_num, and what do the four values add up to each time?")
    mo.vstack([
        mo.md("**One person, other people to compare with**"),
        explore_background,
        mo.md(r"""
        The course's Shapley values for the person aged 41 with education_num 13, made as in the worked example below, with 200 background rows drawn from the people of the training half whose education_num is at most the slider's value. Each bar is one feature's value and starts where the bar above it ends: from the background's average at the top to the person's chance, 0.451, at the bottom.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To share one prediction out over its features:

    1. Take the row to explain, and a background: rows of the data to fill in what is not known.
    2. For each coalition, the features you know, take each background row, put in the known features from the row you explain, ask the model, and average its answers: that is the coalition's value.
    3. For each feature, take every coalition without it, and see what knowing the feature adds to it.
    4. Weigh each addition by the share of the orders in which exactly that coalition comes before the feature, and add them up: that is the feature's Shapley value.
    5. Check that the values add up to the model's answer for the row minus the value with nothing known.
    6. Read each value as a share of that gap, against this background: not as what happens when you change or remove the feature.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The forest of the last lessons with a fourth column, capital_gain, and the same person. `np.random.default_rng(0).choice(n, 200, replace=False)` draws 200 different positions out of `n`. `itertools.product([0, 1], repeat=4)` gives the 16 tuples of four 0s and 1s, from `(0, 0, 0, 0)` to `(1, 1, 1, 1)`: the coalitions. `np.where(condition, a, b)` takes `a` where the condition holds and `b` elsewhere: here the person's value in the known columns and the background row's own value in the others, for all 200 rows at once. `itertools.permutations(range(4))` gives the 24 orders of 0, 1, 2 and 3. Step 4 lets the columns join one by one in each order, and gives each what its joining adds, divided by 24. `f"{x:+.3f}"` writes `x` with its sign and three decimals. Run the cell. It takes a few seconds.
    """)
    return


@app.cell
def example_shapley():
    import itertools

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    # 1. the forest of the last lessons, now with a fourth column, and the person to explain
    _people = pd.read_csv("data/adult.csv.gz")
    _columns = ["age", "education_num", "hours_per_week", "capital_gain"]
    _X = _people[_columns].to_numpy().astype(float)
    _y = (_people["income"] == ">50K").to_numpy()
    _order = np.random.default_rng(0).permutation(len(_y))
    _train, _test = _order[: len(_y) // 2], _order[len(_y) // 2:]
    _forest = RandomForestClassifier(min_samples_leaf=20, random_state=0).fit(_X[_train], _y[_train])
    _person = _X[_test][5]
    print("person:", _person, " chance of yes:", _forest.predict_proba([_person])[0, 1].round(3))

    # 2. the background: 200 people of the training half, to fill in the columns that are not known
    _background = _X[_train][np.random.default_rng(0).choice(len(_train), 200, replace=False)]

    # 3. the value of each of the 16 coalitions: the forest's average chance of yes over the background,
    #    with the known columns (a 1 in the coalition) taken from the person
    _v = {}
    for _known in itertools.product([0, 1], repeat=4):
        _rows = np.where(np.array(_known) == 1, _person, _background)
        _v[_known] = _forest.predict_proba(_rows)[:, 1].mean()
    print("nothing known:", _v[(0, 0, 0, 0)].round(3), " only education_num:", _v[(0, 1, 0, 0)].round(3),
          " all four:", _v[(1, 1, 1, 1)].round(3))

    # 4. what each column adds when it joins, averaged over the 24 orders in which the columns can join
    _orders = list(itertools.permutations(range(4)))
    _values = [0, 0, 0, 0]
    for _join in _orders:
        _known = [0, 0, 0, 0]
        for _i in _join:
            _before = _v[tuple(_known)]
            _known[_i] = 1
            _values[_i] = _values[_i] + (_v[tuple(_known)] - _before) / len(_orders)
    for _i in range(4):
        print("Shapley value of", _columns[_i], f"{_values[_i]:+.3f}")
    print("together:", round(sum(_values), 3),
          " chance minus average:", (_v[(1, 1, 1, 1)] - _v[(0, 0, 0, 0)]).round(3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With nothing known, the forest's average over the background is 0.225; knowing only education_num makes it 0.366, and knowing all four gives 0.451, the person's own chance. education_num gets +0.181 and age +0.087; hours_per_week gets −0.010 and capital_gain −0.032. Together 0.226, exactly the gap: that is Shapley's efficiency. Run the cell again and nothing changes, unlike LIME: there is no random step left once the background is drawn. The formula with its weights gives the same four numbers as the 24 orders; that is exercise 2.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_value():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The value of a coalition

        Write `value(model, person, background, known)`: `model(rows)` gives a list with the model's answer for each row, `person` is the row to explain, `background` is a list of rows, and `known` is a tuple of 0s and 1s, one for each column. For each row of `background`, make a new row `filled` with `person[j]` where `known[j]` is 1, and the background row's own `row[j]` where it is 0. Ask `model` about all the new rows, and return the average of its answers. The check first calls `value(add_up, [4, 0, 1], [[1, 2, 3], [3, 1, 0]], (1, 0, 1))`, where `add_up` gives the sum of each row; then other models, such as `product` and `biggest`. The picture runs your function with the forest for all 16 coalitions of the person.
        """),
        L.task("Write `value(model, person, background, known)`: build a list `rows` with one filled-in row for each background row, and return the average of `model(rows)`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_value():
    box_value = L.code_box("value", lines=14)
    box_value
    return (box_value,)


@app.cell(hide_code=True)
def check_value(box_value):
    result_value = L.check("value", box_value.value)
    result_value
    return (result_value,)


@app.cell(hide_code=True)
def hints_value():
    L.hints(
        "value",
        [
            "A known column comes from the person, an unknown one from the background row. Build one new row for each background row, then ask the model about all of them at once.",
            "Start with `rows = []` and `for row in background:`, with `filled = []` and an inner `for j in range(len(person)):` that appends `person[j]` or `row[j]`. After the loop, `chances = model(rows)`.",
            "`return sum(chances) / ___`",
        ],
        why="`rows = []`, then `for row in background:` with `filled = []`, `for j in range(len(person)):`, and `filled.append(person[j])` if `known[j] == 1`, else `filled.append(row[j])`; then `rows.append(filled)`. After the loop, `chances = model(rows)` and `return sum(chances) / len(chances)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_shapley():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The Shapley value of a column

        The box gives `weight(s, n)`, the weight $s!\,(n-s-1)!/n!$ of a coalition of $s$ columns out of $n$. Finish `shapley(v, i)`: `v` is a dict from each coalition, a tuple of 0s and 1s, to its value, and `i` is the column. Go over the coalitions `known` in `v`. For each one where `known[i]` is 0, make the same coalition with column `i` known: `list(known)`, set place `i` to 1, and `tuple(...)` again to look it up in `v`. Add its weight times the difference of the two values to `total`. A coalition has `sum(known)` columns known, out of `len(known)`. The check first calls `shapley({(0, 0): 0, (1, 0): 2, (0, 1): 4, (1, 1): 10}, 0)`, the pub quiz of *Why this matters*: Ash should get 4. Then games of three and four columns. The picture runs your function on the 16 values of the worked example.
        """),
        L.task("Replace the `...`: start `total = 0`, add for each coalition without column `i` its weight times what `i` adds to it, and return `total`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shapley():
    box_shapley = L.code_box("shapley", lines=17)
    box_shapley
    return (box_shapley,)


@app.cell(hide_code=True)
def check_shapley(box_shapley):
    result_shapley = L.check("shapley", box_shapley.value)
    result_shapley
    return (result_shapley,)


@app.cell(hide_code=True)
def hints_shapley():
    L.hints(
        "shapley",
        [
            "Column `i` adds the value with it minus the value without it to each coalition without it, and each of those differences counts by the weight of the coalition's size: `sum(known)` columns out of `len(known)`.",
            "Start `total = 0` and `for known in v:`. Inside `if known[i] == 0:`, make `with_i = list(known)`, set `with_i[i] = 1`, and add to `total`; after the loop, return `total`.",
            "`total = total + weight(sum(known), len(known)) * (___ - v[known])`",
        ],
        why="`total = 0`, then `for known in v:` with `if known[i] == 0:`, `with_i = list(known)`, `with_i[i] = 1` and `total = total + weight(sum(known), len(known)) * (v[tuple(with_i)] - v[known])`. After the loop, `return total`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_meaning():
    mo.md(r"""
    ### Exercise 3. What the share says
    """)
    return


@app.cell(hide_code=True)
def choice_meaning_cell():
    choice_meaning = L.choice(
        "meaning",
        question="For this person, education_num gets a Shapley value of +0.181. What does that say?",
        options={
            "right": "Its share of the gap to the average",
            "removed": "Without it, the chance would be 0.181 lower",
            "more": "More education raises the chance by 0.181",
        },
        correct="right",
        why={
            "right": "Right. The four values share the gap between the person's chance, 0.451, and the forest's average over the background, 0.225, and education_num's share is +0.181: what knowing it adds, averaged over every order in which the columns can join. Molnar: \"The Shapley value is NOT the difference in prediction when we remove the feature from the model.\"",
            "removed": "Molnar: \"The Shapley value is NOT the difference in prediction when we remove the feature from the model.\" It is an average over the coalitions: knowing education_num adds between 0.139 and 0.226, depending on what else is known. See *What a Shapley value does not say* in *The idea*.",
            "more": "A Shapley value has no direction: \"a positive Shapley value doesn't mean that increasing the feature would increase the prediction\" (Molnar). It says how the gap to the average is shared, for this person as they are. See *What a Shapley value does not say* in *The idea*.",
        },
    )
    choice_meaning
    return (choice_meaning,)


@app.cell(hide_code=True)
def text_exercise_background():
    mo.md(r"""
    ### Exercise 4. Another background
    """)
    return


@app.cell(hide_code=True)
def choice_background_cell():
    choice_background = L.choice(
        "background",
        question="You explain the same person again, with a background of 200 people whose education_num is 13 or more. What happens?",
        options={
            "unique": "Nothing changes: Shapley values are unique",
            "right": "The values change, and so does their sum",
            "only": "Only education_num gets a new value",
        },
        correct="right",
        why={
            "right": "Right. Those people's average chance of yes is 0.444, so the gap to this person's 0.451 is only 0.008 (0.4513 minus 0.4436, before rounding), and education_num gets −0.027: among people with 13 or more, 13 is the lowest. The same person and the same forest, compared with other people. Molnar: \"Shapley value interpretation is always in reference to the dataset that was used for replacing values for absent players.\"",
            "unique": "Shapley's rules make the answer unique for one game, and the background is part of the game: with people whose education_num is 13 or more, the average is 0.444 and education_num gets −0.027. Sundararajan and Najmi: the different ways \"give very different results, rendering the uniqueness result meaningless\". See *Which \"unknown\"?* in *The idea*.",
            "only": "Every feature's share can change, and so can the gap they share: with people whose education_num is 13 or more, the average is 0.444, the four values add up to the gap, 0.008 before rounding, and age gets +0.112 instead of +0.087. See the slider in *The idea*.",
        },
    )
    choice_background
    return (choice_background,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [LIME: a simple model around one prediction](/course/open/rai-lime?lang=en) explained one prediction with the slopes of a line, which change with the kernel width; Shapley values share the prediction out, and change with the background instead. Like LIME they are a local explanation, and model-agnostic: they only need the model's answers, in the terms of [Kinds of explanation](/course/open/rai-kinds-of-explanation?lang=en). The next lesson, [Tools for explanations](/course/open/rai-xai-tools?lang=en), shows the libraries that do this for many features, and which of their choices change the answer.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_background, choice_meaning, result_shapley, result_value):
    L.wrap_up(result_value, result_shapley, choice_meaning, choice_background)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
