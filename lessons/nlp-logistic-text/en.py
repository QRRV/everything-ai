# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-logistic-text. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai.lesson import lesson

    L = lesson("nlp-logistic-text", "en")
    C = L.checks   # the course's words, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en), and one about [Training logistic regression](/course/open/ml-train-logistic?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_tfidf():
    quick_tfidf = L.choice(
        "quick-tfidf",
        question="Of the 2,400 training sentences of this lesson, 170 have *great* and 747 have *and*. A new sentence has *and* twice and *great* once. Which of the two words gets the higher TF-IDF value in that sentence?",
        options={
            "twice": "*and*, since the sentence has it twice",
            "right": "*great*, since fewer sentences have it",
            "zero": "Both get 0, since each is in more than one sentence",
        },
        correct="right",
        why={
            "right": r"*great* gets $1 \times \log_{10}(2400 / 170) = 1.15$; *and* gets $2 \times \log_{10}(2400 / 747) = 1.01$: twice a word that is in almost a third of all sentences counts for less than once a word that is in 7% of them. In this lesson, such values are what logistic regression gets for each sentence.",
            "twice": "The count is only one of the two factors. The IDF of *and* is low, since 747 of the 2,400 sentences have it. See [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en).",
            "zero": "Only a word in every document gets an IDF of 0, and *and* is in 747 of the 2,400 sentences. See [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en).",
        },
    )
    quick_tfidf
    return (quick_tfidf,)


@app.cell(hide_code=True)
def quick_check_slope():
    quick_slope = L.choice(
        "quick-slope",
        question="An example has label 1 and a feature with value 2, and logistic regression gives it the probability 0.8. What is the slope of its log loss for the weight of that feature?",
        options={
            "plus": "0.4, so the step lowers the weight",
            "right": "−0.4, so the step raises the weight",
            "error": "−0.2, the error without the feature value",
            "times": "1.6, the probability times the feature value",
        },
        correct="right",
        why={
            "right": r"The slope is $(p - y) \times x = (0.8 - 1) \times 2 = -0.4$. Gradient descent takes away the learning rate times the slope, so a negative slope raises the weight, and the probability of this positive example goes up. This lesson does the same with one weight for each word.",
            "plus": "With $y = 1$ and $p = 0.8$, the error $p - y$ is negative: the example is positive and gets too low a probability. See [Training logistic regression](/course/open/ml-train-logistic?lang=en).",
            "error": "−0.2 is the error $p - y$ alone; the slope for a weight is the error times the value of its feature. See [Training logistic regression](/course/open/ml-train-logistic?lang=en).",
            "times": "The slope uses the error $p - y$, not the probability itself. See [Training logistic regression](/course/open/ml-train-logistic?lang=en).",
        },
    )
    quick_slope
    return (quick_slope,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In *A document as a vector*, 600 held-out sentences from reviews got the label of the training sentence most like them, and 397 came out right. Naive Bayes, which learns from all 2,400 training sentences how much each word says about the label, got 497 right. This lesson trains the classifier of [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en) on the same sentences, with the gradient descent of [Training logistic regression](/course/open/ml-train-logistic?lang=en). Its features are the TF-IDF values of [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en): one weight for each word, learned from the labels.

    Those weights can be read, and they say what the classifier learned. Daniel Jurafsky and James H. Martin write in *Speech and Language Processing*: "Thus we might expect in a sentiment task the word awesome to have a high positive weight, and abysmal to have a very negative weight." The sentences are the 3,000 of *Sentiment Labelled Sentences*, collected by Dimitrios Kotzias and colleagues: 1,000 each from Amazon, IMDb and Yelp, half of them positive and half negative. As in *A document as a vector*, every fifth sentence is held out, so 2,400 train the model and 600 test it.

    **An everyday example.** A teacher grades an essay with a list of points: a clear opening adds 2, a spelling mistake takes 1 away. The grade is the sum. Logistic regression learns such a list from examples, with a number for each word.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="After training on the 2,400 sentences, which of these four words has the most negative weight?",
        options={
            "bad": "*bad*",
            "worst": "*worst*",
            "right": "*not*",
            "terrible": "*terrible*",
        },
        reveal="***not***, with −1.78. *bad* gets −1.49, *worst* −1.06 and *terrible* −0.85. *not* is in 179 negative training sentences and in 40 positive ones, such as *The last 15 minutes of movie are also not bad as well.*; *bad* is in 72 negative and 2 positive ones. Each sentence pushes the weights of its own words, and the many negative sentences with *not* push its weight down further than the 40 positive ones push it up. *bad* is in only 74 sentences, so far fewer sentences push its weight: already at the start, the slope for *not* is larger, 0.032 against 0.026.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### One weight for each word

    Each sentence becomes a row of TF-IDF values, as in [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en): a dictionary from each word of the sentence to its count times $\log_{10}$ of 2,400 divided by the number of training sentences the word is in. The vocabulary is the 4,518 words of the training sentences; a word that no training sentence has is left out. Logistic regression gives each word $t$ a weight $w_t$, and has one bias $b$. The logit $z$ and the probability $p$ that the sentence is positive are

    $$z = b + \sum_{t} x_t \, w_t \qquad p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

    The sum runs over the words of the sentence, and $x_t$ is the TF-IDF value of word $t$ in it. A word the sentence lacks has value 0 and adds nothing. This is $\mathbf{w} \cdot \mathbf{x} + b$ from [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en), written out for a sparse row: of the 4,518 values of a sentence of ten words, at most ten are not 0. A positive weight pushes $z$ up, towards positive, and a negative weight pushes it down. The picture shows this for a held-out sentence, with the weights after training.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    *I did not expect this to be so good!* is labeled positive, yet gets $p = 0.423$: *not* adds −1.852 and *good* +1.505, and the other words and the bias bring $z$ to −0.309. A weight belongs to one word, whatever stands next to it, so *not* pushes down here as well.

    ### Training: each sentence pushes its own words

    The weights are learned with gradient descent on the mean log loss, as in [Training logistic regression](/course/open/ml-train-logistic?lang=en). Over the $n$ training sentences, the slope for the weight of word $t$ is

    $$\text{slope}(w_t) = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i) \, x_{i,t}$$

    with $p_i$ the probability for sentence $i$, $y_i$ its label, 1 or 0, and $x_{i,t}$ the value of word $t$ in it. This is `grad_w = (diff @ X) / n` from that lesson, with `diff = p - y`, worked out for one word: only the sentences that have word $t$ count, since for the others $x_{i,t} = 0$. A positive sentence that gets too low a probability has a negative error $p_i - y_i$, so the step raises the weights of its words; a negative sentence with too high a probability lowers them. The slope for the bias is the mean error, $\frac{1}{n} \sum_{i} (p_i - y_i)$. Each step moves every weight and the bias by minus the learning rate times its slope.

    ### A classifier with two names

    In NLP, logistic regression has a second name. Jurafsky and Martin tell in their historical notes where it comes from: "At the same time in the early 1990s logistic regression was developed and applied to NLP at IBM Research under the name maximum entropy modeling or maxent (Berger et al., 1996), seemingly independent of the statistical literature." Elsewhere they call maximum entropy "an alternative formulation of logistic regression": another road to the same model. So the **maximum entropy classifier**, maxent for short, is logistic regression under another name.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **maximum entropy classifier** (maxent) | a name in NLP for logistic regression, from the 1990s |
        | **weight of a word** | the number logistic regression learns for a word: positive pushes towards positive, negative towards negative |
        | $x_t$ | the TF-IDF value of word $t$ in a sentence |
        | $w_t$ | the weight of word $t$ |
        """),
        L.watch_out("Two numbers that are both called weight. The TF-IDF value of a word, which [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en) called its weight in a document, comes from counting and is fixed. The weight of logistic regression is learned from the labels. This lesson says TF-IDF value for the first and weight for the second."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Turn each sentence into a row: a dictionary from each of its words to its TF-IDF value.
    2. Start with weight 0 for every word of the vocabulary, and bias 0.
    3. For each training sentence, work out $p$ with the weights as they are now, and its error $p - y$.
    4. For each word, add up the error times the value of the word over the sentences that have it, and divide by the number of sentences: the slope of its weight. The slope of the bias is the mean error.
    5. Move each weight and the bias by minus the learning rate times its slope, and go back to step 3.
    6. Give a new sentence the label positive when its $p$ is 0.5 or more, and count how many held-out sentences get their own label.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The code below takes the held-out sentence *I will not be eating there again.*, labeled negative, and turns it into a row with the course's `row`. The course's `parts` gives, for each word of the row, its TF-IDF value and its weight after training; the code prints both and their product. Then it prints the bias after training, from the course's `bias`.
    """)
    return


@app.cell
def example_cell():
    example = C.row("I will not be eating there again.")
    for word, value, weight in C.parts(example):
        print(word, round(value, 3), round(weight, 3), round(value * weight, 3))
    print("bias", round(C.bias(), 3))
    return (example,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Add it all up: $z = -0.181 + 0.016 + 0.509 - 1.852 - 0.132 - 0.127 - 0.532 + 0.450 = -1.849$, and $p = \sigma(-1.849) = 0.136$. The model says negative, as the label does. *not* does most of the work, *there* helps, and *will* and *again* push the other way, with +0.509 and +0.450.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_probability():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. probability: the probability of positive

        Write `probability(row, weights, bias)`, where `row` is a dictionary from each word of a sentence to its TF-IDF value, `weights` a dictionary from each word to its weight, and `bias` a number. Give back the probability that the sentence is positive: start a sum $z$ at the bias, add for each word of the row its value times its weight, and give back $\sigma(z)$, `1 / (1 + np.exp(-z))`.

        `small_weights`, the course's `SMALL_WEIGHTS`, holds five weights: *great* 2.21, *good* 1.33, *food* −0.16, *not* −1.78 and *bad* −1.49; the examples use the bias −0.18. For `probability({"great": 1.15, "food": 1.363}, small_weights, -0.18)`, $z = -0.18 + 1.15 \times 2.21 + 1.363 \times (-0.16) = 2.143$, and the result is about 0.895. Replace the `...`. The check calls that and `probability({"not": 1.04, "good": 1.13}, small_weights, -0.18)`, then cases of its own: a row without words, a word with a higher value, another bias, and the row of the worked example with `weights` and `bias`, the weights and the bias after training, from the course's `weights` and `bias`.
        """),
        L.task("Work out the probability of positive for one row of TF-IDF values."),
    ])
    return


@app.cell
def weights_cell():
    small_weights = C.SMALL_WEIGHTS
    weights = C.weights()
    bias = C.bias()
    print(small_weights)
    print(len(weights), round(weights["great"], 3), round(weights["food"], 3), round(bias, 3))
    return bias, small_weights, weights


@app.cell(hide_code=True)
def exercise_probability():
    box_probability = L.code_box("probability", lines=5)
    box_probability
    return (box_probability,)


@app.cell(hide_code=True)
def check_probability(box_probability):
    result_probability = L.check("probability", box_probability.value)
    result_probability
    return (result_probability,)


@app.cell(hide_code=True)
def hints_probability():
    L.hints(
        "probability",
        [
            "Start with `z = bias`, and go over the words of the row with `for word in row:`. For each `word`, add `row[word]` times the weight of the word to `z`. After the loop, give back `1 / (1 + np.exp(-z))`.",
            "For the first row, the sum gets 1.15 times 2.21 for *great*. Where in the arguments of the function is 2.21, and how do you look up the number of any word there?",
            r"`        z = z + row[word] * ___`",
        ],
        why=r"`z = bias`; `for word in row:` with `z = z + row[word] * weights[word]`; after the loop, `return 1 / (1 + np.exp(-z))`.",
    )
    return


@app.cell(hide_code=True)
def text_after_probability():
    mo.md(r"""
    In the picture, your function gave seven held-out sentences their probability of positive, with the weights after training. *Everyone is very attentive, providing excellent customer service.* gets 0.94, and *It's like a bad two hour TV movie.* 0.096. *I did not expect this to be so good!*, from the idea, gets 0.423: the wrong label, for a positive sentence.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_gradients():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. gradients: the slopes

        Write `gradients(examples, weights, bias)`, where `examples` is a list of pairs `(row, label)`, with label 1 for positive and 0 for negative. Give back two things: a dictionary from each word of `weights` to the slope of the mean log loss for its weight, and the slope for the bias. Each example adds, for each of its words, its error $p - y$ times the value of the word, divided by $n$, the number of examples; and for the bias its error divided by $n$. Here $p$ is what `probability` gives the row, and the course's `probability` is there for you to use. A word of `weights` that no example has gets slope 0.

        `small_examples`, the course's `SMALL_EXAMPLES`, holds three examples, and `start_weights`, the course's `START_WEIGHTS`, gives the five words weight 0. For `gradients(small_examples, start_weights, 0.0)`, every example gets $p = 0.5$. *great* is only in the first example, which is positive, so its slope is $(0.5 - 1) \times 1.15 / 3 = -0.192$; *food* is in the first and in the third, which is negative, and gets $((0.5 - 1) \times 1.363 + (0.5 - 0) \times 1.363) / 3 = 0$. The result is `({"great": -0.191667, "good": 0.188333, "food": 0.0, "not": 0.173333, "bad": 0.251833}, 0.166667)`. Replace the `...`. The check calls that and `gradients(small_examples, small_weights, -0.18)`, then cases of its own: one example, `more_examples` with another bias, and the 2,400 training sentences, `examples`, from the course's `training_examples`, with every weight 0 in `zero_weights` and with the weights after training. The cell below prints the first training sentence as a row, and as text from the course's `training`.
        """),
        L.task("Work out the slope of the mean log loss for every weight and for the bias."),
    ])
    return


@app.cell
def examples_cell(weights):
    small_examples = C.SMALL_EXAMPLES
    start_weights = C.START_WEIGHTS
    more_examples = C.MORE_EXAMPLES
    examples = C.training_examples()
    zero_weights = {key: 0.0 for key in weights}
    print(small_examples[0])
    print(len(examples), examples[0])
    print(C.training()[0][1])
    return examples, more_examples, small_examples, start_weights, zero_weights


@app.cell(hide_code=True)
def exercise_gradients():
    box_gradients = L.code_box("gradients", lines=10)
    box_gradients
    return (box_gradients,)


@app.cell(hide_code=True)
def check_gradients(box_gradients):
    result_gradients = L.check("gradients", box_gradients.value)
    result_gradients
    return (result_gradients,)


@app.cell(hide_code=True)
def hints_gradients():
    L.hints(
        "gradients",
        [
            "Set `n = len(examples)`, start `grad` with 0.0 for every word of `weights`, and set `grad_bias = 0.0`. Go over the examples with `for row, label in examples:`, and set `error` to the probability of the row minus `label`. Then, for each `word` of the row, add to `grad[word]` the error times the value of the word, divided by `n`, and add `error / n` to `grad_bias`. After the loops, give back `grad, grad_bias`.",
            "The first small example has *great* with value 1.15, and its error at the start is 0.5 − 1 = −0.5. What does that example add to the slope of *great*, before you divide by `n`?",
            r"`            grad[word] = grad[word] + ___ / n`",
        ],
        why=r"`n = len(examples)`, `grad = {word: 0.0 for word in weights}` and `grad_bias = 0.0`; `for row, label in examples:` with `error = probability(row, weights, bias) - label`, then `for word in row:` with `grad[word] = grad[word] + error * row[word] / n`, and after that loop `grad_bias = grad_bias + error / n`; at the end, `return grad, grad_bias`.",
    )
    return


@app.cell(hide_code=True)
def text_after_gradients():
    mo.md(r"""
    In the picture, your function worked out the slopes of the first step, over the 2,400 training sentences, with every weight 0. *great* has the most negative slope, so the first step raises its weight most, and *not* the most positive slope. *and* and *that* are in the lists as well: *and* is in 450 positive and 297 negative training sentences, *that* in 86 positive and 133 negative.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Train it

    The button trains logistic regression on the 2,400 training sentences: 300 steps of gradient descent with learning rate 1, from weight 0 for every word and bias 0. Each step takes the slopes from your `gradients` once exercise 2 is right, and from the course's until then. Each step works out 4,518 slopes, so the 300 steps take some seconds. While it trains, the chart shows the mean log loss of the training sentences after each step; beside it you see the six words with the strongest weight each way, and the probability of positive of three held-out sentences. Afterwards you can go back through the steps.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_button = mo.ui.run_button(label="Train for 300 steps")
    demo_button
    return (demo_button,)


@app.cell(hide_code=True)
def demo_run(box_gradients, demo_button, result_gradients):
    if not demo_button.value:
        _out = mo.md("*Press the button to train.*")
    else:
        _yours = C.mine(box_gradients.value, result_gradients.state == "correct", "gradients", C.EXERCISES["gradients"].names)
        _words = C.live_words(L.lang)
        _now = {}

        def _progress(_step, _loss, _ws, _b):
            if "live" not in _now:
                _now["live"] = L.live({"title": _words["title"], "x": _words["x"], "y": _words["y"], "alt": _words["alt"]},
                                      C.STEPS, low=0.0, high=0.7)
            _now["live"].watch(lambda: C.now_parts(_ws, _b, L.lang))
            _now["live"].add(_step, _loss, _words["progress"].format(s=_step, n=C.STEPS, loss=L.number(_loss, 3)))

        mo.output.replace(mo.md("*Step 1 · starting …*"))
        _run = C.demo(_yours, _progress)
        if isinstance(_run, str):
            _note = {"gradients": "*Your `gradients` stops with an error on the training sentences, or gives something other than a slope for every word and one for the bias. The picture under exercise 2 says what goes wrong.*",
                     "slow": "*The training takes more than two minutes. Does your `gradients` go over the examples only once? Or is the computer busy with something else?*",
                     "course": "*The training takes more than two minutes. Is the computer busy with something else?*"}[_run if _yours else "course"]
            _out = _now["live"].stopped(_note) if "live" in _now else mo.md(_note)
        else:
            _right = C.rewind_right(_run, C.STEPS)
            _who = " Your `gradients` worked out the slopes." if _yours else ""
            _out = mo.vstack([
                mo.md(f"""After {C.STEPS} steps, the mean log loss of the training sentences is {L.number(_run["curve"][-1][1], 3)}, and the model labels <span class="ea-num">{_right}</span> of the 600 held-out sentences right.{_who}"""),
                L.explore("rewind", "step", lambda _s: (C.rewind_frame(_run, int(_s), L.lang),
                                                        f"After step {int(_s)}: {C.rewind_right(_run, int(_s))} of the 600 held-out sentences right."),
                          start=C.STEPS, values=C.LOOKS, role="",
                          task="Go back to step 1, then forward again: which words get a strong weight first, and what happens to *I did not expect this to be so good!*?"),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With every weight 0, each sentence gets $p = 0.5$, and the mean log loss is $\log 2 = 0.693$; after 300 steps it is 0.165. The model then labels 2,354 of the 2,400 training sentences right, 98.1%, and 491 of the 600 held-out sentences, 81.8%: from 456 after the first step to 491 after step 70, and from there between 488 and 491. The strongest words towards positive are *great* (2.21), *good* (1.33), *excellent* (1.16), *nice* (1.14), *love* (1.10) and *works* (0.99); towards negative *not* (−1.78), *bad* (−1.49), *worst* (−1.06), *poor* (−0.99), *terrible* (−0.85) and *don't* (−0.82). Of the two words of Jurafsky and Martin, *awesome* gets 0.74, a high positive weight; *abysmal* is in only two training sentences and gets −0.25: negative, but not very. *and* and *that*, from the picture under exercise 2, end at +0.67 and −0.43.

    *I did not expect this to be so good!* first drops, to 0.393 after step 50: what *not* takes away grows faster than what *good* adds, and *did*, *be* and the bias pull $z$ down too. Then it climbs back a little, to 0.423, as *did* turns positive and *so* grows; *good* never catches up with *not*, so the sentence still does not get its own label.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_accuracy():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. accuracy: how many are right?

        Write `accuracy(examples, weights, bias)`: give each example the label 1 when `probability` gives its row 0.5 or more, and 0 otherwise, count the examples where that is their own label, and give back the share: that count divided by the number of examples.

        For `accuracy(small_examples, small_weights, -0.18)`, the result is 1.0: the three examples get 0.895, 0.371 and 0.066, so the labels 1, 0 and 0, which are their own. `more_examples`, the course's `MORE_EXAMPLES`, holds four examples, and `accuracy(more_examples, small_weights, -0.18)` gives 0.5: two of the four get their own label. Replace the `...`. The check calls those two, then cases of its own: one example that gets the other label, a bias that makes every example positive, and the 600 held-out sentences, `held`, from the course's `held_out_examples`, with the weights after training. The cell below prints the first held-out sentence as a row, and as text from the course's `held_out`.
        """),
        L.task("Count the share of examples that get their own label."),
    ])
    return


@app.cell
def held_cell(more_examples):
    held = C.held_out_examples()
    print(more_examples[0])
    print(len(held), held[0])
    print(C.held_out()[0][1])
    return (held,)


@app.cell(hide_code=True)
def exercise_accuracy():
    box_accuracy = L.code_box("accuracy", lines=9)
    box_accuracy
    return (box_accuracy,)


@app.cell(hide_code=True)
def check_accuracy(box_accuracy):
    result_accuracy = L.check("accuracy", box_accuracy.value)
    result_accuracy
    return (result_accuracy,)


@app.cell(hide_code=True)
def hints_accuracy():
    L.hints(
        "accuracy",
        [
            "Start with `right = 0`, and go over the examples with `for row, label in examples:`. In each round, set `guess = 0`, and set it to 1 when `probability(row, weights, bias)` is high enough. When `guess` equals `label`, add 1 to `right`. After the loop, give back `right / len(examples)`.",
            "The second small example gets the probability 0.371. Which label does the model give it, and from which probability on does it say positive?",
            r"`        if probability(row, weights, bias) >= ___:`",
        ],
        why=r"`right = 0`; `for row, label in examples:` with `guess = 0`, `if probability(row, weights, bias) >= 0.5:` and `guess = 1`, then `if guess == label:` and `right = right + 1`; after the loop, `return right / len(examples)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_accuracy():
    mo.md(r"""
    In the picture, your function measured logistic regression on the 600 held-out sentences: 491 right, 81.8%. Naive Bayes, on the plain counts, gets 497 right, 82.8%: six more. The two do not get the same sentences wrong. Both get the same 83 wrong; only logistic regression gets 26 others wrong, and only naive Bayes 20.

    ### How good is it?

    On these short sentences, logistic regression does no better than naive Bayes. Sida Wang and Christopher Manning found something like it in 2012, when they set naive Bayes against support vector machines, another kind of classifier: "for short snippet sentiment tasks, NB actually does better than SVMs (while for longer documents the opposite result holds)". NB is naive Bayes, and the review sentences of this lesson are short snippets. Andrew Ng and Michael Jordan made a related finding in 2001, about the number of training examples: "while discriminative learning has lower asymptotic error, a generative classifier may also approach its (higher) asymptotic error much faster." The asymptotic error is the error with ever more examples. Naive Bayes is what they call generative: it models how each label makes words. Logistic regression is discriminative: it learns the label straight from the words. So with few examples naive Bayes can be ahead, and with many logistic regression can pass it.

    The weights do help to see why the model decides as it does. Jurafsky and Martin, on interpreting a model: "inspecting its magnitude (how large is the weight w associated with the feature?) can help us interpret why the classifier made the decision it makes." The most similar sentence of *A document as a vector* cannot do that: it only points at one other sentence.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Exercise 4. What weight?

    Each line gives a word, and in how many positive and how many negative training sentences it occurs. Put each word under the weight it gets after the 300 steps of training.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_weight = L.sort("what-weight")
    sort_weight
    return (sort_weight,)


@app.cell(hide_code=True)
def check_sort(sort_weight):
    result_sort = L.check("what-weight", sort_weight)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it: a sentence of your own

    Type a sentence in English. The picture shows what each of its words adds to $z$, with the weights after training, and the probability of positive that follows. Try *Not bad at all.*, or a sentence about a meal, a film or a phone.
    """)
    return


@app.cell(hide_code=True)
def try_controls():
    try_text = mo.ui.text(value=C.DEMO_TEXT, label="Sentence", full_width=True)
    try_text
    return (try_text,)


@app.cell(hide_code=True)
def try_look(try_text):
    mo.Html(C.try_picture(try_text.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_try():
    mo.md(r"""
    *The food was not great, but the staff were lovely.* gets 0.732. *great* adds +2.537 and *not* −1.852: logistic regression adds up what each word says on its own, and cannot see that *not* belongs to *great*. *Not bad at all.* gets 0.008: *not* and *bad* both have a strong negative weight, and *at* takes away another 0.689, so the model is sure, and wrong. A later lesson of the domain, *A list of good and bad words*, turns the score of a word around after a negation.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en) turned each sentence into a row of TF-IDF values; this lesson gave those rows to the logistic regression of [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en), with one weight for each word, trained with the gradient descent of [Training logistic regression](/course/open/ml-train-logistic?lang=en). The weights show which words push towards positive and which towards negative, and on short sentences naive Bayes does about as well. The next lesson of the domain, *Many classes: micro and macro averages*, measures a classifier that chooses among more than two labels.

    **An everyday example.** A spam filter can work in this way: each word of an email has a learned weight, and the sum decides whether the email goes to the spam folder.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_accuracy, result_gradients, result_probability, result_sort):
    L.wrap_up(result_probability, result_gradients, result_accuracy, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
