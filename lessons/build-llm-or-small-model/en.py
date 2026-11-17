# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-llm-or-small-model. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-llm-or-small-model", "en")
    C = L.checks   # the questions of the bank, Qwen3's recorded answers, the course's small model, and the demo with yours


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Ready-made AI: vision, speech, translation and classic NLP](/course/open/build-ai-services?lang=en), one about [The same model with scikit-learn](/course/open/ml-sklearn?lang=en), and one about [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_parse():
    quick_parse = L.choice(
        "quick-parse",
        question="A model answers `Card_Payment_Wrong_Exchange_Rate`. Why does the course's `parse_label` give back the longest label it finds in it?",
        options={
            "piece": "Because `exchange_rate` is in it too, as a piece of the longer label",
            "fast": "Because a longer label is quicker to find",
            "rare": "Because long labels are rarer, so they are more likely right",
        },
        correct="piece",
        why={
            "piece": "Both labels are in the answer; the model meant the longer one.",
            "fast": "Speed has nothing to do with it. Which labels are in that answer? See [Ready-made AI: vision, speech, translation and classic NLP](/course/open/build-ai-services?lang=en).",
            "rare": "How often a label comes up does not matter here. Which labels are in that answer? See [Ready-made AI: vision, speech, translation and classic NLP](/course/open/build-ai-services?lang=en).",
        },
    )
    quick_parse
    return (quick_parse,)


@app.cell(hide_code=True)
def quick_check_fit():
    quick_fit = L.choice(
        "quick-fit",
        question="In scikit-learn, what do `model.fit(X, y)` and `model.predict(X_new)` do?",
        options={
            "same": "Both train the model, the second on new data",
            "both": "`fit` learns from `X` and the right answers `y`; `predict` gives the model's answers for new rows",
            "check": "`fit` checks whether the model fits the data; `predict` trains it",
        },
        correct="both",
        why={
            "both": "Every model of scikit-learn has these two: learn, then use.",
            "same": "`predict` gets no right answers, so what could it learn from? See [The same model with scikit-learn](/course/open/ml-sklearn?lang=en).",
            "check": "It is the other way round: `fit` is where the learning happens. See [The same model with scikit-learn](/course/open/ml-sklearn?lang=en).",
        },
    )
    quick_fit
    return (quick_fit,)


@app.cell(hide_code=True)
def quick_check_f1():
    quick_f1 = L.choice(
        "quick-f1",
        question="A model has precision 1.0 and recall 0.1. What is its F1?",
        options={
            "mean": "0.55, the mean of the two",
            "one": "1.0, the higher of the two",
            "low": "About 0.18, close to the lower one",
        },
        correct="low",
        why={
            "low": "The F1 is the harmonic mean, 2 · 1.0 · 0.1 / 1.1 ≈ 0.18: it stays close to the lower of the two.",
            "mean": "That is the plain mean. Which mean does the F1 take? See [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en).",
            "one": "Then a model that finds almost nothing would score full marks. See [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en).",
        },
    )
    quick_f1
    return (quick_f1,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Ready-made AI: vision, speech, translation and classic NLP](/course/open/build-ai-services?lang=en), Qwen3 sorted questions of customers into six labels with a prompt alone, and got 99 of 120 right. For one fixed task that runs again and again, there is another way: a small model of your own, trained on sorted examples, that does that one task and nothing else. It can be better, it is much faster, and it costs almost nothing per message. But it needs those sorted examples, and it knows only its own task. This lesson measures both on the same 120 questions, so you can choose.

    **An everyday example.** A good cook can make any dish you ask for. A coffee machine makes one thing, but it makes it in seconds, the same every time, and for almost nothing. For coffee, you use the machine; for a new dish, you need the cook. That is only a comparison; below are the real numbers.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **task-specific model** | a small model trained for one task on examples of that task, such as sorting messages into six labels; it knows nothing else |
    | **TF-IDF** | *term frequency, inverse document frequency*: each text becomes a vector with a number for each word the texts use; the number is higher when the word is in this text, and lower when the word is in many texts, such as `my` or `the`; each vector has length 1 |
    | `TfidfVectorizer` | from scikit-learn: `fit_transform(texts)` learns the words of the training texts and makes their vectors; `transform(new_texts)` makes vectors of new texts with those same words |
    | `LogisticRegression` | the logistic regression of [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en), in scikit-learn; with more than two labels, it gives each label a probability, as softmax does, and `predict` picks the most likely |
    | **macro F1** | the F1 score of each label, counting that label as yes and all others as no, and then the mean of those F1 scores: every label counts as much, also one that comes up rarely |
    | `a & b` | with two NumPy arrays of `True` and `False`: `True` where both are `True` |

    ### The small model against Qwen3

    The course trained such a small model, TF-IDF and logistic regression, on sorted questions of the same bank: first on 2 questions of each label, then on 5, 10, 20, 30 and finally on all 50. Each time it measured the macro F1 on the same 120 test questions Qwen3 answered there, which the small model never saw:
    """)
    return


@app.cell(hide_code=True)
def picture_curve():
    mo.Html(C.curve_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_compare():
    mo.vstack([
        mo.md(r"""
        With 2 or 5 questions per label, the small model is worse than Qwen3: it has seen too few words. From 10 per label on it is better, and with all 50 it reaches a macro F1 of 0.934, with 112 of the 120 right, against Qwen3's 0.823 and 99. The line wobbles a little: with 20 test questions per label, one answer more or less right moves the score by about 0.01.

        | | Qwen3-0.6B with a prompt | the small model, 50 per label |
        |---|---|---|
        | macro F1 on the 120 test questions | 0.823 | 0.934 |
        | right of the 120 | 99 | 112 |
        | time for the 120 | 125 seconds on the course's computer | less than a tenth of a second, the training on the 300 questions included |
        | size | 1.5 GB | 32 kB: 443 words, and 6 weights for each |
        | made-up price of a million messages, at 3 dollars per million tokens in and 15 out | 393 dollars: 116 tokens in and 3 out per message | only the computer it runs on |
        | needs | a prompt | 300 sorted questions |

        So for one fixed task with sorted examples, the small model wins on all three. The LLM wins when there are no sorted examples, when the labels change often, when a message needs knowledge of the world or comes in a language the examples do not have, and when the answer is a text instead of a label.
        """),
        L.watch_out("Measure both on the same test texts, and never on texts a model learned from: a small model knows its 300 training questions by heart, and would look better there than it is. The training and test sets of [Will it work on new data? Training and test sets](/course/open/ml-train-test-split?lang=en) are just as needed here."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: a small model for one fixed task

    1. **Collect sorted examples:** texts with their labels, and keep some apart for testing.
    2. **Turn the texts into vectors:** `vectorizer = TfidfVectorizer()`, then `X = vectorizer.fit_transform(texts)`.
    3. **Train:** `model = LogisticRegression(max_iter=1000)`, then `model.fit(X, labels)`. The `max_iter=1000` gives it enough steps to finish learning.
    4. **Sort new texts:** `model.predict(vectorizer.transform(new_texts))`, with `transform`, not `fit_transform`: new texts get vectors with the words of the training texts.
    5. **Measure** the macro F1 on the test set, and compare with the LLM on the same texts.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    What TF-IDF makes of three short texts: `my card is lost`, `my card has not arrived` and `my pin is blocked`.

    1. **The words:** `fit_transform` finds 9 different words, in the order of the alphabet: `arrived`, `blocked`, `card`, `has`, `is`, `lost`, `my`, `not`, `pin`. Each text becomes a vector of 9 numbers.
    2. **The vector of `my card is lost`:** `card` 0.480, `is` 0.480, `lost` 0.632, `my` 0.373, and 0 for the five words that are not in it.
    3. **Why those numbers:** `lost` is in this text only, so it says most about it and gets the highest number; `card` and `is` are in two of the three texts; `my` is in all three, so it says least and gets the lowest. The four numbers make a vector of length 1.
    4. **What the model then learns:** a weight for each word and each label, such as a high weight for `lost` with `lost_or_stolen_card`.
    """)
    return


@app.cell(hide_code=True)
def text_train():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Train the small model

        Finish `train_small(texts, labels)`: it turns the texts into TF-IDF vectors, trains a logistic regression on them, and gives back both, the vectorizer and the model. The check trains your function on the 300 sorted questions, `texts` and `labels`, and on the first 60, and sorts the 120 test questions with what it gives back; the picture puts your model on the course's line.
        """),
        L.task("Make `X` with the vectorizer, and train `model` on `X` and `labels`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_train():
    box_train = L.code_box("train_small", lines=10)
    box_train
    return (box_train,)


@app.cell(hide_code=True)
def check_train(box_train):
    result_train = L.check("train_small", box_train.value)
    result_train
    return (result_train,)


@app.cell(hide_code=True)
def hints_train():
    L.hints(
        "train_small",
        [
            "Steps 2 and 3 of the recipe: first vectors of the texts, then the model learns from those vectors and the labels.",
            "The vectorizer learns the words and makes the vectors in one go, with `fit_transform`; the model learns with `fit`, from the training data and the right labels.",
            "The first `...`: `X = ___`, the vectors of `texts`.",
        ],
        why="""```python
def train_small(texts, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)
    return vectorizer, model
```

The function gives back the vectorizer too: new texts need vectors with the same words before the model can sort them.""",
    )
    return


@app.cell(hide_code=True)
def text_f1():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The macro F1

        `macro_f1(true, predicted, labels)` counts, for each label, the true positives `tp`, false positives `fp` and false negatives `fn` already. Work out the F1 of that label from them, as in the lesson on the F1 score, and give back the mean of all those F1 scores. The check tries a small example, Qwen3's labels, the small model's, and a model that is always right.
        """),
        L.task("Append the F1 of each label to `scores`, and return the mean of `scores`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_f1():
    box_f1 = L.code_box("macro_f1", lines=12)
    box_f1
    return (box_f1,)


@app.cell(hide_code=True)
def check_f1(box_f1):
    result_f1 = L.check("macro_f1", box_f1.value)
    result_f1
    return (result_f1,)


@app.cell(hide_code=True)
def hints_f1():
    L.hints(
        "macro_f1",
        [
            "Two steps: the F1 of one label inside the loop, the mean of all of them after it.",
            "The F1 from the counts is twice `tp`, divided by twice `tp` plus `fp` plus `fn`; `np.mean` gives the mean of a list.",
            "In the loop: `scores.append(2 * tp / (___))`; after it, the mean of `scores`.",
        ],
        why="""```python
def macro_f1(true, predicted, labels):
    true, predicted = np.array(true), np.array(predicted)
    scores = []
    for label in labels:
        tp = np.sum((true == label) & (predicted == label))
        fp = np.sum((true != label) & (predicted == label))
        fn = np.sum((true == label) & (predicted != label))
        scores.append(2 * tp / (2 * tp + fp + fn))
    return np.mean(scores)
```

Each label is yes once, and the rest no; the mean makes every label count as much.""",
    )
    return


@app.cell(hide_code=True)
def text_when():
    mo.md(r"""
    ### Exercise 3. When does the LLM win?
    """)
    return


@app.cell(hide_code=True)
def choice_when_cell():
    choice_when = L.choice(
        "when-llm",
        question="A shop wants to sort the reviews of its products by topic. The topics change every month, and nobody has time to sort examples. What fits?",
        options={
            "small": "A small model trained on last month's sorted reviews",
            "nothing": "Nothing: without examples no model can sort",
            "llm": "An LLM with a prompt that lists this month's topics",
        },
        correct="llm",
        why={
            "llm": "A prompt changes in a minute and needs no examples; a small model would need new sorted examples every month.",
            "small": "Last month's topics are not this month's. What would the small model have to learn the new ones from?",
            "nothing": "Qwen3 sorted 99 of 120 questions without a single example. What did it get instead?",
        },
    )
    choice_when
    return (choice_when,)


@app.cell(hide_code=True)
def text_test_set():
    mo.md(r"""
    ### Exercise 4. Why the test questions?
    """)
    return


@app.cell(hide_code=True)
def choice_test_set_cell():
    choice_test_set = L.choice(
        "test-set",
        question="Why does the course measure the small model on the 120 test questions, and not on the 300 it learned from?",
        options={
            "seen": "On the 300 it would look better than it is, because it has seen those already",
            "more": "Because 120 questions are more than 300",
            "faster": "Because measuring on fewer questions is quicker",
        },
        correct="seen",
        why={
            "seen": "A model can learn its training questions by heart; only questions it never saw show how it does on the next message of a customer.",
            "more": "120 is fewer than 300. What is different about the 120?",
            "faster": "Both take less than a second. What is different about the 120?",
        },
    )
    choice_test_set
    return (choice_test_set,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Train your small model on fewer or more questions

    Once exercise 1 is right, your `train_small` learns here from the first questions of each label, as many as the slider says, and you see where it lands against Qwen3. Write a message of your own too, and see which label it gets. Try a message in Dutch: the small model knows only the words of its English questions.
    """)
    return


@app.cell(hide_code=True)
def slider_cell():
    n_slider = mo.ui.slider(start=2, stop=50, step=1, value=10, label="Sorted questions per label", show_value=True)
    message_box = mo.ui.text(value="Someone took my card out of my bag.", label="A message of your own", full_width=True)
    mo.vstack([n_slider, message_box])
    return message_box, n_slider


@app.cell(hide_code=True)
def demo_answer(box_train, message_box, n_slider):
    _train = C.learner_train(box_train.value)
    if _train is None:
        _out = mo.md("*Finish exercise 1 first: the demo trains your own `train_small`.*")
    else:
        _words = {"title": "Your small model, on the first questions of each label",
                  "note": "Trained on {n} questions per label, {total} in all, in {s} seconds: macro F1 {f1} on the 120 test questions (the circle), against Qwen3's {q}.",
                  "label": "Your message gets the label",
                  "failed": "Your train_small stopped with an error: {why}"}
        _out = mo.Html(C.demo(_train, n_slider.value, message_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The 120 questions and Qwen3's answers come from [Ready-made AI: vision, speech, translation and classic NLP](/course/open/build-ai-services?lang=en); `fit` and `predict` from [The same model with scikit-learn](/course/open/ml-sklearn?lang=en), and the F1 score from [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en). The price is counted as in [Tokens and prices: count before you send](/course/open/build-tokens-prices?lang=en). Later lessons of the module bring the model to your own device and into the browser, where a small model fits best.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_test_set, choice_when, result_f1, result_train):
    L.wrap_up(result_train, result_f1, choice_when, choice_test_set)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
