# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-project-spam. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("nlp-project-spam", "en")
    C = L.checks   # the messages, the course's filter and naive Bayes, the demos and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Logistic regression on words](/course/open/nlp-logistic-text?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_tf_idf():
    quick_tf_idf = L.choice(
        "quick-tf-idf",
        question="A word is twice in a message, and in 10 of the 1,000 training messages. What is its TF-IDF value, with the logarithm with base 10?",
        options={
            "idf": "2",
            "right": "4",
            "no-log": "200",
            "upside": "0.02",
        },
        correct="right",
        why={
            "idf": "That is the inverse document frequency alone, $\\log_{10}(1000/10) = 2$. The TF-IDF value multiplies it by how often the word is in the message: $2 \\times 2 = 4$. See [Logistic regression on words](/course/open/nlp-logistic-text?lang=en), which made its rows from TF-IDF values.",
            "right": "The count, 2, times $\\log_{10}(1000/10) = 2$. In this project, every message gets such a value for each of its words.",
            "no-log": "That is $2 \\times 1000/10$, without the logarithm. The inverse document frequency is $\\log_{10}(1000/10) = 2$, so the value is $2 \\times 2 = 4$.",
            "upside": "That divides the other way round, $10/1000$, and leaves out the logarithm. The inverse document frequency is $\\log_{10}(1000/10) = 2$, so the value is $2 \\times 2 = 4$.",
        },
    )
    quick_tf_idf
    return (quick_tf_idf,)


@app.cell(hide_code=True)
def quick_check_step():
    quick_step = L.choice(
        "quick-step",
        question="A weight is 0.4, and its slope is 0.2. What is the weight after one step of gradient descent with learning rate 1.0?",
        options={
            "plus": "0.6",
            "right": "0.2",
            "times": "0.08",
            "same": "0.4",
        },
        correct="right",
        why={
            "plus": "That adds the slope. Gradient descent goes against the slope, downhill: $0.4 - 1.0 \\times 0.2 = 0.2$. See the demo of [Logistic regression on words](/course/open/nlp-logistic-text?lang=en).",
            "right": "$0.4 - 1.0 \\times 0.2 = 0.2$: the weight moves against its slope, by the learning rate times the slope. In this project you write that step yourself.",
            "times": "That multiplies the weight by the slope. A step subtracts the learning rate times the slope: $0.4 - 1.0 \\times 0.2 = 0.2$. See the demo of [Logistic regression on words](/course/open/nlp-logistic-text?lang=en).",
            "same": "Only a slope of 0 leaves a weight where it is. Here the step subtracts $1.0 \\times 0.2$, so the weight becomes 0.2.",
        },
    )
    quick_step
    return (quick_step,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        In [Logistic regression on words](/course/open/nlp-logistic-text?lang=en) you trained logistic regression on 2,400 sentences from reviews, with the course's code for the training. In this project you build a text classifier of your own, piece by piece, and its first job is to filter spam: real text messages, put together in a collection in 2011, with real spam among them. The functions you write stay in your project. Its next step, *Project: sentiment in English and Dutch*, uses them again on reviews.

        A spam filter makes two kinds of mistakes. It can let spam through, and it can block a message that is not spam. The first costs a glance. The second can cost much more: a message from a friend, a doctor or a bank that never shows up. One number for how often a filter is right does not say which mistakes it makes, so your project counts both kinds apart.

        **An everyday example.** Someone who sorts the post of an office throws away the advertising. A folder for a pizza place that lands on a desk costs a colleague a glance; a letter from the tax office that lands in the bin can cost a fine. So whoever sorts throws a letter away only when they are sure, and lets a few folders through. That is only a comparison; below you count what your filter does.
        """),
        draw.project_steps(["Your spam filter", "Your sentiment classifier"], 0,
                           here_label="← today", title="Where your project stands: text"),
    ])
    return


@app.cell(hide_code=True)
def predict_through():
    guess_through = L.guess(
        "through",
        question="A filter that blocks nothing lets every message through. How many of the 1,034 test messages of this project does it get right, do you think?",
        options={
            "spam": "About 13%",
            "half": "About 50%",
            "right": "About 87%",
            "all": "Almost 100%",
        },
        reveal="**About 87%: 895 of the 1,034.** Only 139 of the test messages are spam, so a filter that does nothing is right about all the others. How often a filter is right says little when one label is rare. Your project counts the spam it lets through and the messages it blocks, each on its own.",
    )
    guess_through
    return (guess_through,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **The messages.** The SMS Spam Collection, put together by Tiago A. Almeida and José María Gómez Hidalgo in 2011, holds 5,574 text messages in English, each labeled spam or not. The 747 spam messages come from Grumbletext, a British forum where people complained about spam they got, and from an earlier collection by Gómez Hidalgo. The other 4,827 come from that collection too, from volunteers at the National University of Singapore, most of them students, and from a thesis about text messages by Caroline Tagg. The collection may be used under the license CC BY 4.0, but the messages that are not spam are real personal messages, with first names and at times a phone number, so the course does not host it: in the worked example you download it once from the UCI Machine Learning Repository. The lesson shows only messages it chose, with long numbers masked as [number].

    **Copies, and the split.** 403 lines of the file are a copy of an earlier line: the same spam sent to many people, or a short message such as *Sorry, I'll call later*, which is there 30 times. If one copy lands among the messages to train on and another among the test messages, the filter has seen that test message before, and the test makes the filter look better than it is. So your project splits the 5,171 different texts: every fifth one is a test message. That gives 4,137 messages to train on, 514 of them spam, and 1,034 test messages, 139 of them spam. Texts that differ only in digits, signs or capital letters count as different, but have the same words: 24 test messages, 19 of them spam, have exactly the same words as a message to train on, most of them the same spam in a slightly different form. Such spam comes back on real phones too, so the course leaves those messages in.

    **Your filter.** It is the classifier of [Logistic regression on words](/course/open/nlp-logistic-text?lang=en): a TF-IDF row for every message, and one weight for every word. Two things are new. The document frequencies come from the messages to train on only, since the test messages stand for messages that arrive later. And a word of a test message that no message to train on has gets no column: the filter learned no weight for it. 493 of the 1,034 test messages have such a word. Before training, every weight is 0 and every message gets $p = 0.5$; each step of gradient descent moves every weight and the bias against its slope, as in [Logistic regression on words](/course/open/nlp-logistic-text?lang=en).

    **Two kinds of mistakes.** The filter gives every message a probability of spam, $p$, and blocks it when $p$ is at the **threshold** or above. In [Logistic regression on words](/course/open/nlp-logistic-text?lang=en) that was 0.5. A **missed spam message** is spam that the filter lets through; a **wrongly blocked message** is one that is not spam and that the filter blocks. A higher threshold blocks fewer messages, so fewer are blocked wrongly, and more spam gets through; a lower threshold does the opposite. Where to put it depends on what each mistake costs.

    **Something to compare with.** *Spam or not: naive Bayes* classified messages by counting how often each word comes in spam and in other messages. Your project trains naive Bayes on the same messages to train on, with the same words, and compares.

    The picture follows one test message through the filter, with the weights after training, as you will train it below.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        *voicemail* is in none of the 4,137 messages to train on, so it gets no column and adds nothing. *call* adds the most, 0.992, and *please* and *new* add some too, but the bias of −3.344 keeps $z$ below 0: $p$ is 0.253, below 0.5, so the filter lets this spam through. The same message with *message* in place of *voicemail*, which is in 54 messages to train on, gets 0.409.

        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **threshold** | the lowest probability of spam at which the filter blocks a message | `threshold` |
        | **missed spam message** | a spam message that the filter lets through | `missed` |
        | **wrongly blocked message** | a message that is not spam, and that the filter blocks | `blocked` |
        """),
        L.watch_out("Look for copies before you split a dataset. Had your project put every fifth line of the file among the test messages, 120 of those 1,114 test messages would also be among the messages to train on, with exactly the same text."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Your spam filter:

    1. Download the messages, and split the different texts: every fifth one is a test message.
    2. Count the document frequencies on the messages to train on. That is `document_frequencies`.
    3. Make the TF-IDF row of every message, with no column for a word that the messages to train on lack. That is `features`.
    4. Start every word at weight 0 and the bias at 0, and take 200 steps of gradient descent with learning rate 1.0. That is `step`, which `train` calls.
    5. At a threshold, count on the test messages the spam let through and the messages blocked wrongly. That is `mistakes`.
    6. Choose the threshold from what each mistake costs, and compare with naive Bayes.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell below follows steps 1 and 2. `sms` is the course's `everything_ai.sms`, which the cell under the button gives once the messages are downloaded. `sms.training()` gives the messages to train on as pairs of a list of words and a label, 1 for spam and 0 for not spam, and `sms.test()` gives the test messages. The words are in small letters, as in [Logistic regression on words](/course/open/nlp-logistic-text?lang=en); digits and signs are left out. The cell counts the spam among the messages to train on, then the document frequencies: of every message, each different word once, with `set(words)`. Last, it prints the document frequency of five words. Run the cell.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download the messages, 0.2 MB")
    return (get_button,)


@app.cell(hide_code=True)
def sms_cell(get_button):
    sms = C.sms_or_stop(get_button, {
        "ask": "*The SMS Spam Collection is not on your computer yet. The button downloads it once from the UCI Machine Learning Repository, 0.2 MB. The worked example and the demos need it; the exercises work without it. The course does not host it: the messages that are not spam are real personal messages.*",
        "busy": "**Downloading the messages** · {done} of {total} KB",
        "no_net": "*The download did not work: the UCI Machine Learning Repository could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 0.2 MB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
        "here": "*The SMS Spam Collection is on your computer: 5,574 text messages, by Tiago A. Almeida and José María Gómez Hidalgo, under CC BY 4.0.*",
    })
    return (sms,)


@app.cell
def example_counts(sms):
    # 1. the messages to train on and the test messages: (words, label), label 1 for spam
    _training = sms.training()
    _test = sms.test()
    _spam = 0
    for _words, _label in _training:
        _spam = _spam + _label
    print(len(_training), "messages to train on,", _spam, "of them spam;", len(_test), "test messages")

    # 2. the document frequencies: in how many messages to train on each word is
    _df = {}
    for _words, _label in _training:
        for _word in set(_words):
            if _word not in _df:
                _df[_word] = 0
            _df[_word] = _df[_word] + 1
    print(len(_df), "different words")
    for _word in ["call", "free", "i", "p", "lt"]:
        print(_word, _df[_word])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    6,957 different words: your filter will have 6,957 weights. *call* is in 367 messages to train on, 225 of them spam, and *i* in 1,289, only 34 of them spam. Two words look odd. *p* is in 121 messages, 114 of them spam: spam writes prices as *150p*, 150 pence, and without the digits a *p* is left. *lt* is in 181 messages, none of them spam: in many messages that are not spam, `&lt;#&gt;` stands where a number was taken out, the way a web page writes `<#>`, and its words are *lt* and *gt*. A filter learns whatever tells the labels apart in its data, even traces of how the data was put together. Exercise 1 makes the counts of step 2 and the TF-IDF rows of step 3 into functions of your project.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_features():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. features: the row of a message

        `features(words, df, n)` gets the words of one message, the document frequencies `df` of the messages to train on, and their number `n`. It gives the TF-IDF row of the message as a dictionary: for every word of the message that `df` knows, how often it is in the message times its inverse document frequency, with the logarithm with base 10, as in *TF-IDF: weighing the words*. A word that `df` does not know gets no column. The box already has `document_frequencies`, the loop of the worked example as a function, and the start of `features`.

        `small_df`, the course's `SMALL_DF`, holds the document frequencies of eight small messages of the course, such as *win a free prize call now*, and `small_words`, the course's `SMALL_WORDS`, is the message *free prize free call tonight*. *free* is in 4 of the 8 small messages, *prize* and *call* in 3, and *tonight* in none. So `features(small_words, small_df, 8)` gives `{'free': 0.602, 'prize': 0.426, 'call': 0.426}`: *free* twice $\log_{10}(8/4)$, the others once $\log_{10}(8/3)$, and no column for *tonight*. Replace the `...`. The check calls that, then cases of its own: a message with only words that `df` does not know, three words with document frequencies of the real messages to train on, and a message without words. When it is right, the check saves both functions in your project, in `work/projects/text/features.py`.
        """),
        L.task("Give every word of the message that `df` knows its TF-IDF value, and skip the others."),
    ])
    return


@app.cell(hide_code=True)
def exercise_features():
    box_features = L.code_box("features", lines=24)
    box_features
    return (box_features,)


@app.cell(hide_code=True)
def check_features(box_features):
    result_features = L.check("features", box_features.value)
    result_features
    return (result_features,)


@app.cell(hide_code=True)
def hints_features():
    L.hints(
        "features",
        [
            "Step 3 of the recipe. Start `row` as an empty dictionary and loop over `words`. Skip a word that is not in `df`. The first time a word comes along, give `row[word]` the value `0.0`; each time, add to it.",
            "Each time a word comes along, its row gets one more inverse document frequency, so a word that is twice in the message ends with twice that value. Which NumPy function takes the logarithm with base 10, and of which two numbers divided?",
            "`            row[word] = row[word] + ___`",
        ],
        why="`row = {}`; `for word in words:` with `if word in df:`, under it `if word not in row:` with `row[word] = 0.0`, and `row[word] = row[word] + np.log10(n / df[word])`; after the loop, `return row`. With the real document frequencies, *call* twice gets $2 \\times \\log_{10}(4137/367) = 2.104$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. step: one step of training

        `step(examples, weights, bias, rate)` takes one step of gradient descent, as the course's code did in [Logistic regression on words](/course/open/nlp-logistic-text?lang=en). `examples` are TF-IDF rows with their labels. `gradients`, from the file `work/projects/text/logistic.py` that the course put in your project, gives `grad`, the slope of every weight, and `grad_bias`, the slope of the bias. Each weight and the bias take one step against their slope, with `rate` as the learning rate. The box already has the first lines of `step`, and `train`, which starts every word of the examples at weight 0 and the bias at 0, and calls your `step` `steps` times.

        `small_examples`, the course's `SMALL_EXAMPLES`, holds three small TF-IDF rows with the words *free*, *prize*, *call* and *txt*, two of them spam, and `start_weights` gives those four words weight 0. With every weight 0, every example gets $p = 0.5$. *prize* is only in the two spam examples and gets the slope $-0.142$, so its new weight is $0 - 1.0 \times (-0.142) = 0.142$. `step(small_examples, start_weights, 0.0, 1.0)` gives `({'free': 0.050, 'prize': 0.142, 'call': -0.071, 'txt': 0.100}, 0.167)`. Replace the `...`. The check calls that, then cases of its own: other weights, another bias and another learning rate, and a weight for a word that no example has. It saves `step` and `train` in `work/projects/text/train.py`.
        """),
        L.task("Make the new weights and the new bias of one step of gradient descent."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("step", lines=24)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("step", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "step",
        [
            "`gradients` already gave `grad`, with a slope for every word, and `grad_bias`. Loop `for word in weights:` and give `new_weights[word]` the new value of each weight; after the loop, make `new_bias`.",
            "A step goes against the slope, downhill: a weight with a negative slope goes up. How big the step is, `rate` decides. What do you subtract from `weights[word]`?",
            "`        new_weights[word] = weights[word] - ___`",
        ],
        why="`for word in weights:` with `new_weights[word] = weights[word] - rate * grad[word]`, and after the loop `new_bias = bias - rate * grad_bias`. In the second case, with learning rate 0.5 from the small weights and bias −0.6, *prize* goes from 2.0 to 2.036.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Train your filter

    The button trains your filter on the 4,137 messages to train on. Your `document_frequencies` and `features` make the rows, and your `step` takes 200 steps with learning rate 1.0, from weight 0 for every word and bias 0. A file that is not in your project yet, such as `train.py` before exercise 2 is right, is replaced by the course's version. The chart shows the mean log loss of the messages to train on after each step. Beside it you see the five words with the strongest weight each way, three test messages with their probability of spam, and how many test messages the filter gets wrong at 0.5. It takes about ten seconds. Afterwards you can go back through the steps.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_button = mo.ui.run_button(label="Train for 200 steps")
    demo_button
    return (demo_button,)


@app.cell(hide_code=True)
def demo_run(demo_button, result_features, result_step, sms):
    _r = (result_features, result_step, sms)   # so that this cell runs again after each check
    trained = None
    if not demo_button.value:
        _out = mo.md("*Press the button to train.*")
    else:
        _words = C.live_words(L.lang)
        _now = {}

        def _progress(_step, _loss, _ws, _b, _test_rows):
            if "live" not in _now:
                _now["live"] = L.live({"title": _words["title"], "x": _words["x"], "y": _words["y"], "alt": _words["alt"]},
                                      C.STEPS, low=0.0, high=0.7)
            _now["live"].watch(lambda: C.live_parts(_test_rows, _ws, _b, L.lang))
            _now["live"].add(_step, _loss, _words["progress"].format(s=_step, n=C.STEPS, loss=L.number(_loss, 3)))

        mo.output.replace(mo.md("*Making the rows of the messages …*"))
        _run = C.demo(_progress)
        if isinstance(_run, str):
            _note = C.broken(_run, L.lang)
            _out = _now["live"].stopped(_note) if "live" in _now else mo.md(_note)
        else:
            trained = _run
            _m, _bl = C.rewind_counts(_run, C.STEPS)
            _who = {(): "the course's functions", ("features",): "your `features` and the course's `step`",
                    ("train",): "the course's `features` and your `step`"}.get(
                tuple(m for m in _run["fns"].yours if m != "judge"), "your `features` and your `step`")
            _out = mo.vstack([
                mo.md(f"""After {C.STEPS} steps with {_who}, the mean log loss of the messages to train on is {L.number(_run["curve"][-1][1], 3)}. At 0.5, the filter lets <span class="ea-num">{_m}</span> of the 139 test spam messages through and blocks <span class="ea-num">{_bl}</span> of the 895 others."""),
                L.explore("rewind", "step", lambda _s: (C.rewind_frame(_run, int(_s), L.lang),
                                                        "After step {s}: {m} of the 139 spam messages let through, {b} of the 895 others blocked.".format(
                                                            s=int(_s), m=C.rewind_counts(_run, int(_s))[0], b=C.rewind_counts(_run, int(_s))[1])),
                          start=C.STEPS, values=C.LOOKS, role="",
                          task="Go back to step 1, then forward again: when does *Please CALL [number] immediately* cross 0.5, and which words are strong first?"),
            ])
    _out
    return (trained,)


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With every weight 0, every message gets $p = 0.5$, and the mean log loss is $\log 2 = 0.693$; after 200 steps it is 0.031, and the filter gets 4,114 of the 4,137 messages to train on right. At 0.5 it lets 15 of the 139 test spam messages through, and blocks none of the 895 others. The strongest words towards spam are *call* (0.94), *p* (0.93), *txt* (0.73), *free* (0.71), *text* (0.71) and *stop* (0.66); towards not spam *i* (−0.81), *my* (−0.50), *gt* and *lt* (both −0.42), *me* (−0.41) and *ok* (−0.38). *p* and *lt* are the odd words of the worked example. The bias ends at −3.344, lower than the share of spam alone would give: with 514 spam messages among 4,137, that share would start every message at $p = 0.124$, and the bias starts it at 0.034. Spam messages have many words with large weights that pull them over 0.5, so the filter can start every message far from spam.

    Of the three test messages, *Please CALL [number] immediately as there is an urgent message waiting for you* ends at 0.712 and is blocked. *You have 1 new voicemail. Please call [number]*, the message of the picture, ends at 0.253 and gets through. *Hey now am free you can call me.* is not spam: it has *free* and *call*, but its words add only 1.29 together, against the bias of −3.344, so it ends at 0.114.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_mistakes():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. mistakes: the two kinds of mistakes

        `mistakes(examples, weights, bias, threshold)` counts, among rows with their labels, the spam messages that the filter lets through, `missed`, and the messages that are not spam and that it blocks, `blocked`. The filter blocks a message whose probability of spam, from the course's `probability` in your `logistic.py`, is `threshold` or more. The box already has the loop and `p`.

        `more_examples`, the course's `MORE_EXAMPLES`, holds the three small examples and three more, and `small_weights` gives *free* 1.2, *prize* 2.0, *call* −0.8 and *txt* 1.5. With bias −0.6, the six examples get the probabilities 0.726, 0.359, 0.760, 0.281, 0.660 and 0.380; the second and the fifth are not spam. At 0.5, the fourth and the sixth, both spam, get through, and the fifth is blocked: `mistakes(more_examples, small_weights, -0.6, 0.5)` gives `(2, 1)`. Replace the `...`. The check calls that, then cases of its own: the thresholds 0.3 and 0.8, and two messages whose probability is exactly the threshold. It saves your function in `work/projects/text/judge.py`.
        """),
        L.task("Count the spam let through and the other messages blocked, at the threshold `threshold`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mistakes():
    box_mistakes = L.code_box("mistakes", lines=14)
    box_mistakes
    return (box_mistakes,)


@app.cell(hide_code=True)
def check_mistakes(box_mistakes):
    result_mistakes = L.check("mistakes", box_mistakes.value)
    result_mistakes
    return (result_mistakes,)


@app.cell(hide_code=True)
def hints_mistakes():
    L.hints(
        "mistakes",
        [
            "Two `if` lines, each with the label, `p` and `threshold` in it: spam is `label == 1`, a message that is not spam `label == 0`. Under each, add 1 to `missed` or to `blocked`.",
            "Spam gets through when its `p` is below the threshold. A message that is not spam is blocked when its `p` is the threshold or more, also when it is exactly the threshold. Which comparison is that?",
            "`        if label == 0 and p ___ threshold:`",
        ],
        why="`if label == 1 and p < threshold:` with `missed = missed + 1` under it, and `if label == 0 and p >= threshold:` with `blocked = blocked + 1`. At 0.3, the second example, with 0.359, is blocked too, and of the spam only the fourth gets through: `(1, 2)`.",
    )
    return


@app.cell(hide_code=True)
def text_project_demo():
    mo.md(r"""
    ### What your project can do now

    Your project now has all three parts: the rows, the training and the count of mistakes. The slider moves the threshold of the filter you trained above, from 0.05 to 0.95. The picture shows every test message as a dot at its probability of spam: spam on the upper row, the others on the lower one. Under it are the three test messages of the training and the test message that is not spam with the highest probability. The sentence gives what your `mistakes` counts at that threshold, beside naive Bayes, trained on the same messages to train on.
    """)
    return


@app.cell(hide_code=True)
def project_demo(result_mistakes, trained):
    _r = result_mistakes   # so that this cell runs again after the check of exercise 3
    if trained is None:
        _out = mo.md("*Train your filter above first: the slider moves its threshold.*")
    else:
        _out = L.explore("threshold", "threshold", lambda _t: C.threshold_frame(trained, float(_t), L.lang),
                         start=0.5, values=C.THRESHOLDS, role="",
                         task="Find the lowest threshold at which your filter blocks none of the 895 test messages that are not spam. How much spam gets through there?")
    _out
    return


@app.cell(hide_code=True)
def text_after_project_demo():
    mo.vstack([
        mo.md(r"""
        At 0.5, your filter lets 15 spam messages through and blocks none; naive Bayes lets 10 through and blocks 5. Both get 1,019 of the 1,034 test messages right, with different mistakes. Lower the threshold, and your filter catches more spam: at 0.35 it lets 11 through and still blocks none. At 0.3 it lets 10 through and blocks one, *Please protect yourself from e-threats. SIB never asks for sensitive information…*: a warning from a bank, which a filter should let through. At 0.1 it lets 2 through, but blocks 16. Naive Bayes blocks, among others, *Just taste fish curry :-P*: the smiley gives the word *p*, which is in 114 spam messages to train on and in 7 others.
        """),
        L.watch_out("Here you chose a threshold by looking at the test messages, so their numbers now flatter your filter: the threshold fits these 1,034 messages. To choose one fairly, set part of the messages to train on aside for it, and look at the test messages once, at the end."),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_phone():
    mo.md(r"""
    ### Exercise 4. Which filter for a phone?
    """)
    return


@app.cell(hide_code=True)
def choice_phone_cell():
    choice_phone = L.choice(
        "phone",
        question="At 0.5, your filter lets 15 of the 139 test spam messages through and blocks none of the 895 others. Naive Bayes lets 10 through and blocks 5. Both get 1,019 right. Which would you put on a phone, for its text messages?",
        options={
            "nb": "Naive Bayes, since less spam gets through",
            "right": "Yours, since a blocked message may go unread",
            "same": "Either, since both get 1,019 right",
            "neither": "Neither, since a filter on a phone has to catch all spam",
        },
        correct="right",
        why={
            "nb": "Five spam messages fewer cost five glances fewer. But naive Bayes buys that with five blocked messages that are not spam, such as *Just taste fish curry :-P*, which the person may never see. On a phone, that is the costlier mistake.",
            "right": "A spam message that gets through costs a glance; a blocked message from a friend, a doctor or a bank can cost much more, and nobody notices it is missing. So for a phone, the filter that blocks nothing wrongly wins, at the price of a few more spam messages. The slider showed that a lower threshold can catch more spam, and that 0.3 already blocks a bank's warning.",
            "same": "The same number right hides different mistakes: 15 spam messages through against 10 through and 5 others blocked. What each mistake costs decides, and on a phone a blocked message costs more.",
            "neither": "No filter here catches all spam without blocking other messages: at 0.05, your filter still lets one spam message through, and it blocks 53 others. A filter weighs the two kinds of mistakes; it does not get rid of them.",
        },
    )
    choice_phone
    return (choice_phone,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Logistic regression on words](/course/open/nlp-logistic-text?lang=en) gave you the classifier, its gradients and the steps of its training. Here it became a filter of your own, on real messages, with what a project needs around it: a split without copies, document frequencies counted on the messages to train on only, training code of your own, and a count of the two kinds of mistakes. *Spam or not: naive Bayes* gave something to compare with, and how often a filter is right turned out to say little when spam is rare. Your project `text` now has `logistic.py`, `features.py`, `train.py` and `judge.py`. It comes back in *Project: sentiment in English and Dutch*, where the same functions classify reviews. The next lesson, *Tools and corpora for Dutch*, turns to the tools and the collections of text that exist for Dutch.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_phone, result_features, result_mistakes, result_step):
    L.wrap_up(result_features, result_step, result_mistakes, choice_phone)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
