# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-cross-entropy. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow log-loss

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-cross-entropy", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Three short questions about [Many classes: softmax](/course/open/ml-softmax?lang=en), [Scoring probabilities with the log: log loss](/course/open/ml-log-loss?lang=en) and [One-hot vectors for classes](/course/open/ml-one-hot?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_softmax():
    quick_softmax = L.choice(
        "quick-softmax",
        question="Two classes both get the logit 0. What does softmax give?",
        options={
            "zero": "0 and 0",
            "right": "0.5 and 0.5",
            "one": "1 and 1",
        },
        correct="right",
        why={
            "zero": "Softmax takes $e$ to the power of each logit first. What is $e^0$? See *The idea* in [Many classes: softmax](/course/open/ml-softmax?lang=en).",
            "one": "$e^0 = 1$ for both, but then softmax divides by their sum. See *The idea* in [Many classes: softmax](/course/open/ml-softmax?lang=en).",
            "right": "$e^0 = 1$ for both, divided by their sum, 2: equal logits give equal probabilities.",
        },
    )
    quick_softmax
    return (quick_softmax,)


@app.cell(hide_code=True)
def quick_check_log_loss():
    quick_log_loss = L.choice(
        "quick-log-loss",
        question="A mail is spam, $y = 1$, and the model gives $\\hat{y} = 0.8$. About how big is its log loss?",
        options={
            "other": "1.61",
            "negative": "−0.22",
            "right": "0.22",
        },
        correct="right",
        why={
            "negative": "The log of 0.8 is negative. What does log loss do with it? See *The idea* in [Scoring probabilities with the log: log loss](/course/open/ml-log-loss?lang=en).",
            "other": "That is $-\\log(0.2)$, the loss if the mail were not spam. See *The idea* in [Scoring probabilities with the log: log loss](/course/open/ml-log-loss?lang=en).",
            "right": "For $y = 1$ the log loss is $-\\log(\\hat{y}) = -\\log(0.8) \\approx 0.22$.",
        },
    )
    quick_log_loss
    return (quick_log_loss,)


@app.cell(hide_code=True)
def quick_check_one_hot():
    quick_one_hot = L.choice(
        "quick-one-hot",
        question="There are four classes, 0 to 3. What is the one-hot vector of class 2?",
        options={
            "right": "(0, 0, 1, 0)",
            "from-one": "(0, 1, 0, 0)",
            "number": "(0, 0, 2, 0)",
        },
        correct="right",
        why={
            "from-one": "That has its 1 at place 1. Where does counting start? See *The idea* in [One-hot vectors for classes](/course/open/ml-one-hot?lang=en).",
            "number": "A one-hot vector holds only 0s and one 1. See *The idea* in [One-hot vectors for classes](/course/open/ml-one-hot?lang=en).",
            "right": "Places 0, 1, 2 and 3: the 1 stands at place 2.",
        },
    )
    quick_one_hot
    return (quick_one_hot,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Softmax gives a probability for every class. To train a model, you need one number that says how good those probabilities are: a loss. For two classes that was log loss. With many classes, the model spreads its probability over all of them: which of those numbers should the loss look at, and how? Cross-entropy is the answer, and it is the loss of nearly every model that picks one class out of many: one that reads digits, one that names what is in a photo, and a language model, which picks its next word out of tens of thousands.

    **An everyday example.** In a quiz, you spread 10 points over the answers of each question, to show how sure you are. Then the quizmaster has to turn your spread into one score per question. Which points should count?
    """)
    return


@app.cell(hide_code=True)
def predict_dog():
    guess_dog = L.guess(
        "dog",
        question="A model sees a photo of a dog. It gives cat 0.7, dog 0.2 and bird 0.1. What do you think is its loss?",
        options={
            "best": "About 0.36: minus the log of 0.7",
            "right": "About 1.61: minus the log of 0.2",
            "all": "About 4.27: minus the logs of all three, added up",
        },
        reveal="**About 1.61.** Only the probability of the right class counts, and the photo is of a dog: $-\\log(0.2) \\approx 1.61$. The 0.7 for cat costs nothing by itself, but it is why dog gets only 0.2. Had the model given dog 0.7, the loss would have been $-\\log(0.7) \\approx 0.36$. As in the quiz, only the points on the right answer count.",
    )
    guess_dog
    return (guess_dog,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A model gives $K$ probabilities $\hat{y}_0, \dots, \hat{y}_{K-1}$, one per class, with the classes numbered from 0 as for one-hot vectors. The label $c$ is the right class, and as a one-hot vector $y_0, \dots, y_{K-1}$ it has a 1 at place $c$ and 0 everywhere else. **Cross-entropy** is minus the sum, over the classes, of each $y_k$ times the log of $\hat{y}_k$:
        """),
        L.formula(r"[[loss:\mathcal{L}]] = -\sum_{k=0}^{K-1} y_k \log([[pred:\hat{y}_k]]) = -\log([[pred:\hat{y}_c]])"),
        mo.md(r"""
        Every $y_k$ is 0, except the one at the right class, so only one term is left: minus the log of the probability of the right class. That is the surprise of what really happened, as in log loss. A probability of 1 for the right class costs 0; the smaller it is, the more it costs. For many examples at once, the loss is the mean of their cross-entropies.

        With two classes it is log loss itself. Say the label is $y$, 1 or 0, and the model gives class 1 the probability $\hat{y}$ and class 0 the rest, $1 - \hat{y}$. The one-hot label is then $(1 - y, y)$, and the sum has two terms: $-\left((1 - y) \log(1 - \hat{y}) + y \log(\hat{y})\right)$. That is the log loss of one example: $-\log(\hat{y})$ for a label 1, and $-\log(1 - \hat{y})$ for a label 0.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **cross-entropy** | | the loss for many classes: minus the log of the probability the model gives the right class; for many examples, the mean of that | `loss` |
        | $c$ | c | the right class of an example: its label | `label` |
        | $y_k$, $\hat{y}_k$ | y k, y hat k | place $k$ of the one-hot label, and the probability the model gives class $k$ | `onehot[k]`, `p[k]` |
        """),
        L.watch_out("Cross-entropy only looks at the probability of the right class, but the others still matter: all probabilities add up to 1, so more for a wrong class leaves less for the right one. Two models with the same probability for the right class get the same loss, however they spread the rest."),
        mo.md(r"""
        Move the slider: it sets the logit of dog, while cat keeps the logit 2 and bird the logit 0. The photo is of a dog, class 1. One chart shows the <span class="c-pred">probabilities</span> from softmax, dog's with a heavier outline; the other the <span class="c-loss">cross-entropy</span>, minus the log of dog's probability.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_z_cell():
    def _frame(z):
        ps = C.probabilities(z)
        shown = [L.number(p, 2) for p in ps]
        dog = L.number(ps[C.RIGHT], 4)
        loss = L.number(C.loss(z), 2)
        return (C.picture(z, {"bars": "Probabilities from softmax", "class": "class: 0 cat, 1 dog, 2 bird", "p": "probability",
                              "curve": "Cross-entropy", "p_right": "probability of dog", "loss": "loss",
                              "alt": f"Three bars for cat, dog and bird: {shown[0]}, {shown[1]} and {shown[2]}. The curve of minus the log of dog's probability, with a point at {dog}: a loss of {loss}."}, shown),
                f"""The photo is of a dog, and the model gives dog <span class="c-pred ea-num">{dog}</span>. The cross-entropy is minus the log of that: <span class="c-loss ea-num">{loss}</span>.""")

    explore_z = L.explore("logit-dog", "z₁", _frame, start=0, low=-3, high=5, step=0.5, role="", task="From which logit of dog is the cross-entropy below 0.1? What is dog's probability then?")
    explore_z
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    From a logit of 4.5 for dog, the cross-entropy is below 0.1: dog's probability is then 0.9148, and the loss 0.09. It never reaches 0, however high the logit: that would take a probability of exactly 1.

    ### Recipe

    To work out the cross-entropy of many examples at once, with the probabilities in the rows of `probs` and the labels in `labels`:

    1. Turn the labels into one-hot rows: `onehot = np.eye(K)[labels]`.
    2. Take the log of every probability: `np.log(probs)`.
    3. Multiply by the one-hot rows and add up each row: only the log of the right class is left. Put a minus in front: `-(onehot * np.log(probs)).sum(axis=1)`.
    4. Take the mean over the examples: `.mean()`.
    5. Check one example by hand: minus the log of the probability of its right class.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Three photos and three classes: cat (0), dog (1) and bird (2). The photos are of a dog, a cat and a bird. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_cross_entropy():
    import numpy as np

    _probs = np.array([[0.2, 0.7, 0.1], [0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
    _labels = np.array([1, 0, 2])
    # 1. one-hot rows
    _onehot = np.eye(3)[_labels]
    print(_onehot)
    # 2. the log of every probability
    _logs = np.log(_probs)
    print(_logs)
    # 3. only the right class is left in each row, with a minus in front; then add up each row
    _picked = -(_onehot * _logs)
    print(_picked)
    _losses = _picked.sum(axis=1)
    print(_losses)
    # 4. the mean over the photos
    print(_losses.mean())
    # 5. the first photo by hand: a dog, with probability 0.7
    print(-np.log(0.7))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Each one-hot row keeps one log of its row, and the minus makes it positive: 0.357 for the dog, 0.511 for the cat and 0.357 for the bird, with 0 everywhere else. Added up per row, the photos cost 0.357, 0.511 and 0.357, and their mean is 0.408. The cat costs most: the model gave it only 0.6. The last line gives 0.357 again, the same as the first photo.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_no_idea():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. No idea
    """)
    return


@app.cell(hide_code=True)
def choice_no_idea_cell():
    choice_no_idea = L.choice(
        "no-idea",
        question="A model with four classes has no idea: it gives each class 0.25. What is its cross-entropy, whatever the right class is?",
        options={
            "zero": "0: it says nothing wrong",
            "right": "About 1.39",
            "four": "About 5.55: 1.39 for each of the four classes",
        },
        correct="right",
        why={
            "zero": "Is 0.25 for the right class good? Only a probability of 1 costs 0. See *The idea*.",
            "four": "Which classes count in cross-entropy? See the formula in *The idea*.",
            "right": "Only the right class counts, and that gets 0.25: $-\\log(0.25) = \\log(4) \\approx 1.39$. A model that gives all $K$ classes the same probability has the cross-entropy $\\log(K)$, whatever the label.",
        },
    )
    choice_no_idea
    return (choice_no_idea,)


@app.cell(hide_code=True)
def text_exercise_cross_entropy():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Cross-entropy as a function

        Finish `cross_entropy(p, label)`: it gets the probabilities of one example, as a list, and the number of its right class, and gives back its cross-entropy. Replace the `...`. The check calls `cross_entropy([0.7, 0.2, 0.1], 1)`, the photo of the dog from the guess: that should give about 1.6094. It also tries other probabilities and labels.
        """),
        L.task("Make `cross_entropy` give back the cross-entropy of the probabilities `p` and the label `label`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cross_entropy():
    box_cross_entropy = L.code_box("cross-entropy", lines=6)
    box_cross_entropy
    return (box_cross_entropy,)


@app.cell(hide_code=True)
def check_cross_entropy(box_cross_entropy):
    result_cross_entropy = L.check("cross-entropy", box_cross_entropy.value)
    result_cross_entropy
    return (result_cross_entropy,)


@app.cell(hide_code=True)
def hints_cross_entropy():
    L.hints(
        "cross-entropy",
        [
            "Only the probability of the right class counts. Which element of `p` is that?",
            "`label` is the place of the right class in `p`. What turns a probability into its surprise, and which sign does that need?",
            "`return ___(p[label])`.",
        ],
        why="`p[label]` is the probability of the right class, and `-np.log` turns it into its surprise. For `[0.7, 0.2, 0.1]` and the label 1: $-\\log(0.2) \\approx 1.61$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_batch():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Four photos at once

        Four photos, three classes. The probabilities of each photo are a row of `probs`, and their labels are in `labels`; `onehot` already holds them as one-hot rows. Put the mean cross-entropy of the four photos in `loss`. Replace the `...`. The last line is `loss`, so the check sees it. The check also tries other labels.
        """),
        L.task("Put the mean cross-entropy of `probs` and `onehot` in `loss`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_batch():
    box_batch = L.code_box("batch-loss", lines=8)
    box_batch
    return (box_batch,)


@app.cell(hide_code=True)
def check_batch(box_batch):
    result_batch = L.check("batch-loss", box_batch.value)
    result_batch
    return (result_batch,)


@app.cell(hide_code=True)
def hints_batch():
    L.hints(
        "batch-loss",
        [
            "Steps 2 to 4 of the recipe, in one line: `onehot` is step 1, already done.",
            "Multiply `onehot` by `np.log(probs)`. Then add up along each row, and take the mean of what comes out. Where does the minus go?",
            "`loss = -(onehot * np.log(probs)).sum(axis=___).mean()`.",
        ],
        why="`onehot * np.log(probs)` keeps the log of the right class in each row, `.sum(axis=1)` gives one number per photo, the minus turns them into surprises, and `.mean()` averages them: about 0.83.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Many classes: softmax](/course/open/ml-softmax?lang=en) gave the probabilities, and [One-hot vectors for classes](/course/open/ml-one-hot?lang=en) the labels as vectors; cross-entropy brings them together in one number. It is the surprise from [Scoring probabilities with the log: log loss](/course/open/ml-log-loss?lang=en), for any number of classes, and with two classes it is log loss. In the next lesson, *The gradient of softmax with cross-entropy*, you find how the loss changes with each logit, so that gradient descent can train a model for many classes.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_no_idea, result_cross_entropy, result_batch):
    L.wrap_up(choice_no_idea, result_cross_entropy, result_batch)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
