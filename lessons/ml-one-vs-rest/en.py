# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-one-vs-rest. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-one-vs-rest", "en")
    C = L.checks   # the penguins of ml-knn, the course's one-vs-rest and one-vs-one streets, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [The widest street: the maximum-margin classifier](/course/open/ml-max-margin?lang=en) and [Many classes: softmax](/course/open/ml-softmax?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_side():
    quick_side = L.choice(
        "quick-side",
        question="A soft-margin street between the Chinstraps ($y = -1$) and the Gentoos ($y = +1$) gives a new penguin $w \\cdot x + b = -0.4$. What does it say?",
        options={
            "right": "Chinstrap: the penguin is on the street, but on the Chinstrap side of the line",
            "none": "Nothing: a penguin on the street gets no class",
            "gentoo": "Gentoo: it is closer to the line than to the Chinstrap edge of the street",
        },
        correct="right",
        why={
            "none": "The street is for training: it says how much room the line has. To classify, only the side of the line counts, and $-0.4$ is below 0. See *The idea* in [The widest street: the maximum-margin classifier](/course/open/ml-max-margin?lang=en).",
            "gentoo": "Closer to the line means less sure, not the other class: below 0 is the Chinstrap side, however close. See *The idea* in [The widest street: the maximum-margin classifier](/course/open/ml-max-margin?lang=en).",
            "right": "Below 0 is the Chinstrap side. The score also says how sure the model is: $-0.4$ is not very sure. Today you compare such scores between models.",
        },
    )
    quick_side
    return (quick_side,)


@app.cell(hide_code=True)
def quick_check_softmax():
    quick_softmax = L.choice(
        "quick-softmax",
        question="A model with softmax gives three classes the logits $-1.5$, $-0.2$ and $-0.9$. Which class does it predict?",
        options={
            "none": "None: every logit is below 0",
            "right": "The second: the highest logit gets the highest probability",
            "first": "The first: the logit furthest from 0",
        },
        correct="right",
        why={
            "none": "Softmax turns any logits into probabilities that add up to 1, negative ones too; the highest logit always gets the highest probability. See *The idea* in [Many classes: softmax](/course/open/ml-softmax?lang=en).",
            "first": "The furthest from 0 is the lowest here, and the lowest logit gets the lowest probability. See *The idea* in [Many classes: softmax](/course/open/ml-softmax?lang=en).",
            "right": "$-0.2$ is the highest, so the second class gets the highest probability, though every logit is below 0. Today the highest score wins in the same way.",
        },
    )
    quick_softmax
    return (quick_softmax,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The widest street splits two classes, the Chinstraps from the Gentoos. But the penguins come in three species, handwritten digits in ten, and the maximum-margin classifier only ever answers "this one or that one". Many models do. A model with softmax gives every class its own logit, all in one model, but not every model can be turned into that. What you can always do is train several models for two classes and combine their answers. There are two common ways to do that, and they do not always agree.

    **An everyday example.** In a round-robin tournament every pair of teams plays once, and the team with the most wins takes the cup. No match ever has more than two teams in it.
    """)
    return


@app.cell(hide_code=True)
def predict_models():
    guess_models = L.guess(
        "models",
        question="One way trains one model for every pair of classes. How many models is that for the ten digits 0 to 9?",
        options={
            "ten": "10: one for every digit",
            "twenty": "20: two for every digit",
            "right": "45",
        },
        reveal="**45**: 0 against 1 to 9 is 9 pairs, 1 against 2 to 9 is 8 more, and so on: 9 + 8 + … + 1 = 45, or 10 · 9 / 2. One model per class would be 10. For 100 classes it is 4,950 against 100; but a model for a pair trains only on the examples of its two classes.",
    )
    guess_models
    return (guess_models,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **One-vs-rest.** Train one model for every class, on all the examples: that class $+1$, every other class $-1$. For the penguins: Adelie or not, Chinstrap or not, Gentoo or not. Each model gives a score $w_c \cdot x + b_c$, and above 0 it says yes. By the signs alone, a penguin can get no yes at all, or more than one. So **one-vs-rest** does not look at the signs: it takes the class whose model gives the highest score, the model that is most sure, even when every score is below 0. Put the $w_c$ side by side as the columns of a matrix $W$ and the $b_c$ in a vector $b$, and $x\,W + b$ gives all the scores at once:
        """),
        L.formula(r"\hat{y} = \text{the class } c \text{ with the highest } (x\,[[param:W]] + [[param:b]])_c"),
        mo.md(r"""
        With $k$ classes that is $k$ models, each trained on all the examples.

        **One-vs-one.** Train one model for every pair of classes, only on the examples of those two: Adelie against Chinstrap, Adelie against Gentoo, Chinstrap against Gentoo. Each model gives one vote, to the one of its two classes it picks. In **one-vs-one** the class with the most votes wins. With $k$ classes that is $k(k-1)/2$ models: 3 for three species, 45 for ten digits; but each sees only a part of the data.

        **Output code.** A third way takes one-vs-rest further. Give every class a row of $+1$ and $-1$, its code, and train one model for every column: it learns $+1$ for the classes with $+1$ in its column. A new example gets a row of answers from the models, and takes the class whose code differs from it in the fewest places. One-vs-rest is the code with one $+1$ in every row, read by the scores instead of the signs; one-vs-one fits too if a code may also hold $0$, for a class a model never sees. With more columns than needed, a mistake by one model can be outvoted by the others.

        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **one-vs-rest** | a model for every class, that class against all the others; the class with the highest score wins | `ovr_predict(names, W, b, x)` |
        | **one-vs-one** | a model for every pair of classes, trained on those two only; the class with the most votes wins | `ovo_votes(x, names, pairs)` |
        | **output code** | a row of $+1$ and $-1$ for every class, a model for every column; the class whose row differs from the answers in the fewest places wins | |
        | $W$ | the weights of the one-vs-rest models, one column for every class | `W` |
        """),
        L.watch_out("The three one-vs-rest models are trained apart, so their scores do not share a scale the way the probabilities of softmax do. Taking the highest works well in practice, but nothing guarantees it. And one-vs-one can end in a tie: with three classes, the three models can each vote for a different class. Here the first class in `names` then wins, as with `np.argmax`; in the picture that happens on 0.02% of the plane."),
        mo.md(r"""
        The picture: the penguins of the training set by bill length and body mass, Adelies (circles), Chinstraps (triangles) and Gentoos (squares), and the streets of this lesson, each with $C$ = 1. Left, one-vs-rest: dashed, the line of each of its three models, where its score is 0; <span class="c-pred">in blue</span>, where the highest score changes species. Right, one-vs-one: dashed, the line of each pair's model; in blue, where the most votes change species.
        """),
    ])
    return


@app.cell(hide_code=True)
def figure_ways():
    mo.Html(C.ways_picture({"ovr": "One-vs-rest", "ovo": "One-vs-one", "x": "bill length (mm)", "y": "body mass (g)",
                            "alt": "Two charts of the training penguins by bill length and body mass: Adelies at the bottom left, Chinstraps at the bottom right, Gentoos at the top. In each, three dashed lines, the lines of the three models, and three blue boundaries that meet near 45 mm and 4,200 g and split the plane into a part for each species. On the left, one-vs-rest, the blue boundaries do not follow the dashed lines; on the right, one-vs-one, they are pieces of them. The two sets of blue boundaries lie close together."}))
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    A penguin of your own: drag the dot over the plane, or move it with the arrow keys. The chart beside it shows the penguins of the training set, the boundaries of one-vs-rest in blue and those of one-vs-one dashed; the text gives the score of every one-vs-rest model and the votes of one-vs-one.
    """)
    return


@app.cell(hide_code=True)
def plane_penguin_cell():
    _names = ["Adelie", "Chinstrap", "Gentoo"]
    _yes = ["no model says yes", "one model says yes", "two models say yes", "all three say yes"]

    def _frame(_b, _m):
        _s, _v = C.scores(_b, _m), C.votes(_b, _m)
        _n = sum(1 for _x in _s if _x > 0)
        _r, _o = C.ovr_says(_b, _m), C.ovo_says(_b, _m)
        _said = (f"A penguin of {_b} mm and {C.grouped(_m, L.lang)} g. The scores of one-vs-rest: "
                 + ", ".join(f"{_a} {L.number(_x, 2)}" for _a, _x in zip(_names, _s))
                 + f"; {_yes[_n]}. One-vs-rest says **{_r}**. The votes of one-vs-one: "
                 + ", ".join(f"{_a} {_x}" for _a, _x in zip(_names, _v))
                 + f". One-vs-one says **{_o}**." + (" They disagree." if _r != _o else ""))
        _alt = (f"The training penguins with the boundaries of one-vs-rest and one-vs-one, and a new penguin at {_b} mm and "
                f"{C.grouped(_m, L.lang)} g: one-vs-rest says {_r}, one-vs-one says {_o}.")
        return C.plane_picture(_b, _m, {"title": "Your penguin and the two ways", "x": "bill length (mm)",
                                         "y": "body mass (g)", "alt": _alt}), _said

    _plane = L.plane("penguin", "bill length (mm)", "mass (g)", _frame, C.BILLS, C.MASSES, start=(40, 3600), role="",
                     task="Find a penguin for which no one-vs-rest model says yes, and one for which the two ways disagree.")
    _plane
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    In the middle, between the three species, no one-vs-rest model says yes: at 44 mm from 3,600 to 4,400 g, at 46 mm and 4,000 or 4,400 g, and at 48 mm and 4,400 g, a penguin is "the rest" to all three models. One-vs-rest still answers, with the highest score, and at each of these stops one-vs-one gives the same answer. At the edges two models often say yes: at 34 mm and 5,200 g both the Adelie model (4.15) and the Gentoo model (1.68). Among these stops the two ways disagree only twice, at 38 mm and 5,200 g (one-vs-rest Adelie, one-vs-one Gentoo) and at 54 mm and 4,800 g (Gentoo against Chinstrap), both off the clouds of penguins in the chart. Where the models saw few examples, nothing told them what to say.

    ### Recipe

    To classify many classes with models for two:

    1. Train a model for every class against the rest (one-vs-rest), or for every pair of classes on the examples of those two (one-vs-one).
    2. One-vs-rest: the score of every model, `x @ W + b`, and the class of the highest, `np.argmax`.
    3. One-vs-one: every pair's model gives one vote, to its second class for a score above 0 and to its first otherwise; the class with the most votes wins.
    4. Compare the ways on the validation set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A Gentoo from the validation set: 45.8 mm and 4,200 g, standardized as in the k-NN lesson. The cell holds the course's three one-vs-rest models and three one-vs-one models, each a soft-margin street with $C$ = 1, and asks both ways about this penguin. The `#` comments say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_ways():
    import numpy as np

    # the Gentoo of 45.8 mm and 4,200 g, standardized as in the k-NN lesson
    x = np.array([0.361, 0.058])
    names = ["Adelie", "Chinstrap", "Gentoo"]

    # 1. one-vs-rest: a column of W and a number of b for every species, that species against the other two
    W = np.array([[-2.509, 2.648, 0.392], [-0.146, -2.337, 2.709]])
    b = np.array([-0.264, -1.743, -1.229])

    # 2. the score of every model, and the species of the highest
    scores = x @ W + b
    print(scores)
    print(names[np.argmax(scores)])

    # 1. one-vs-one: a model for every pair, trained on those two species only; above 0 is a vote for the second
    pairs = [("Adelie", "Chinstrap", np.array([2.335, -1.301]), -0.804),
             ("Adelie", "Gentoo", np.array([1.759, 1.813]), -0.433),
             ("Chinstrap", "Gentoo", np.array([-1.416, 2.293]), 0.765)]

    # 3. the score of every pair's model
    for first, second, w, b_pair in pairs:
        print(first, second, round(x @ w + b_pair, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    All three scores are below 0: no model says this penguin is its species. The highest, $-0.923$, is the Chinstrap model's, less than 0.01 above the Gentoo model's: one-vs-rest says Chinstrap, and is wrong. The pairs: Adelie against Chinstrap gives $-0.037$, a vote for Adelie; Adelie against Gentoo gives 0.307 and Chinstrap against Gentoo 0.387, two votes for Gentoo. One-vs-one says Gentoo, and is right. On all 68 penguins of the validation set, one-vs-rest gets 4 wrong and one-vs-one 3; they disagree on 3 penguins, and on the training set both get 14 of the 206 wrong.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ovr():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The highest score

        `ovr_predict(names, W, b, x)` must give the species one-vs-rest says for one penguin `x`, as step 2 of the recipe. Replace the two `...`. The check calls `ovr_predict(names, W, b, np.array([-1.0, -0.4]))` with the `names`, `W` and `b` of the worked example, and tries five more: penguins for which no model or two models say yes, four made-up classes, and three measurements. The picture under the check runs your function on all 68 penguins of the validation set.
        """),
        L.task("Let `ovr_predict` give back the species of the model with the highest score."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ovr():
    box_ovr = L.code_box("ovr-predict", lines=6)
    box_ovr
    return (box_ovr,)


@app.cell(hide_code=True)
def check_ovr(box_ovr):
    result_ovr = L.check("ovr-predict", box_ovr.value)
    result_ovr
    return (result_ovr,)


@app.cell(hide_code=True)
def hints_ovr():
    L.hints(
        "ovr-predict",
        [
            "Step 2 of the recipe, as in the worked example: first a score for every model, then the species of the highest.",
            "`x @ W` gives one number for every column of `W`, one for every model; add `b`. `np.argmax` gives the position of the highest, and `names[...]` the species at that position.",
            "`scores = x @ W + b` and `return names[np.___(scores)]`.",
        ],
        why="For `np.array([-1.0, -0.4])` the scores are 2.303, −3.456 and −2.705: only the Adelie model says yes, and it has the highest score.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_ovo():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Count the votes

        `ovo_votes(x, names, pairs)` must count the votes of the one-vs-one models for one penguin `x`, as step 3 of the recipe. `pairs` is a list like the one in the worked example: `(first, second, w, b)` for every pair of classes. The first lines give every class 0 votes; replace the `...` in the loop. The check calls `ovo_votes(np.array([0.361, 0.058]), names, pairs)` with the Gentoo and the `pairs` of the worked example, and tries five more: three other penguins, four made-up classes with six pairs, and three measurements.
        """),
        L.task("Let every pair's model give one vote, and `ovo_votes` give back the votes of every class."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ovo():
    box_ovo = L.code_box("ovo-votes", lines=10)
    box_ovo
    return (box_ovo,)


@app.cell(hide_code=True)
def check_ovo(box_ovo):
    result_ovo = L.check("ovo-votes", box_ovo.value)
    result_ovo
    return (result_ovo,)


@app.cell(hide_code=True)
def hints_ovo():
    L.hints(
        "ovo-votes",
        [
            "Step 3 of the recipe. Each pair's model gives exactly one vote: work out its score, and see whether it is above 0.",
            "An `if` with the score `x @ w + b > 0`, and an `else`. A vote for `second` is `votes[second] = votes[second] + 1`.",
            "`if x @ w + b > 0:` then `votes[second] = votes[second] + 1`, and `else:` then `votes[___] = votes[___] + 1`, each on its own line, indented under the `for`.",
        ],
        why="Adelie against Chinstrap gives this penguin −0.037: a vote for Adelie. The other two pairs give 0.307 and 0.387: two votes for Gentoo.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_code():
    mo.md(r"""
    ### Exercise 3. A wrong model outvoted

    An output code for the three species with five models. Each row is the code of a species: the answer every model learns for it. Model 1 learns $+1$ for the Adelies and $-1$ for the other two; model 4 learns $+1$ for the Adelies and the Gentoos.

    | | model 1 | model 2 | model 3 | model 4 | model 5 |
    |---|---|---|---|---|---|
    | Adelie | $+1$ | $-1$ | $-1$ | $+1$ | $+1$ |
    | Chinstrap | $-1$ | $+1$ | $-1$ | $-1$ | $+1$ |
    | Gentoo | $-1$ | $-1$ | $+1$ | $+1$ | $-1$ |

    For a new penguin, the five models answer $-1$, $-1$, $-1$, $+1$, $+1$.
    """)
    return


@app.cell(hide_code=True)
def choice_code_cell():
    choice_code = L.choice(
        "output-code",
        question="Which species does the output code give this penguin?",
        options={
            "right": "Adelie: its row differs from the answers in one place only",
            "not-adelie": "Chinstrap or Gentoo: model 1, the one that learns $+1$ for the Adelies only, says $-1$",
            "none": "None: no row matches the answers exactly",
        },
        correct="right",
        why={
            "not-adelie": "An output code never trusts one model alone: it compares the whole row. Count the places where each row differs from the answers. See *Output code* in *The idea*.",
            "none": "With five models a row will often not match exactly; you take the nearest row, the one that differs in the fewest places. See *Output code* in *The idea*.",
            "right": "Adelie differs only at model 1, Chinstrap at models 2 and 4, Gentoo at models 3 and 5. Model 1 was wrong, and the other four outvoted it. Every two codes here differ in at least three places, so with one wrong model the right row is still the nearest: an error-correcting output code.",
        },
    )
    choice_code
    return (choice_code,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The models for two classes are the streets of [The widest street: the maximum-margin classifier](/course/open/ml-max-margin?lang=en). Taking the highest score is the choice of [Many classes: softmax](/course/open/ml-softmax?lang=en), but there one model learns all the scores together, so they share one scale. A model that handles many classes by itself, such as k-NN from [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en), needs neither way. In the next lesson, *No model wins everywhere*, you put the models of this module side by side: each wins where what it assumes about the data holds.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_code, result_ovo, result_ovr):
    L.wrap_up(result_ovr, result_ovo, choice_code)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
