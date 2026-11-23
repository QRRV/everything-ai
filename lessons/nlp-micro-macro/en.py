# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-micro-macro. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-micro-macro", "en")
    C = L.checks   # the course's words, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en), and one about [Logistic regression on words](/course/open/nlp-logistic-text?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_f1():
    quick_f1 = L.choice(
        "quick-f1",
        question="A spam filter finds 30 of the 50 spam mails, and calls 12 other mails spam as well. What is its F1 score for spam?",
        options={
            "recall": "0.6",
            "right": "0.65",
            "precision": "0.71",
            "mean": "0.657",
        },
        correct="right",
        why={
            "right": r"$2 \times 30 / (2 \times 30 + 12 + 20) = 60 / 92 = 0.652$: between the precision, 0.714, and the recall, 0.6, and closer to the smaller one. This lesson works out such an F1 score for each of 59 labels.",
            "precision": "0.71 is the precision: 30 of the 42 mails called spam. The F1 score also counts the 20 spam mails the filter misses. See [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en).",
            "recall": "0.6 is the recall: 30 of the 50 spam mails. The F1 score also counts the 12 mails wrongly called spam. See [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en).",
            "mean": r"0.657 is the plain mean of the precision, 0.714, and the recall, 0.6. The F1 score is $2PR / (P + R)$, which lies closer to the smaller of the two: 0.652. See [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en).",
        },
    )
    quick_f1
    return (quick_f1,)


@app.cell(hide_code=True)
def quick_check_sentence():
    quick_sentence = L.choice(
        "quick-sentence",
        question="In [Logistic regression on words](/course/open/nlp-logistic-text?lang=en), *I did not expect this to be so good!*, a sentence labeled positive, got the probability 0.423 of positive. With positive as the positive label, what is this sentence for the classifier?",
        options={
            "tp": "A true positive, since the sentence is positive",
            "fp": "A false positive, since its probability is too low",
            "right": "A false negative: positive, but called negative",
            "tn": "A true negative, since 0.423 is below 0.5",
        },
        correct="right",
        why={
            "right": "With 0.423, below 0.5, the classifier says negative while the label is positive: a false negative. In this lesson each label in turn is the positive one, and one mistake is then a false negative for one label and a false positive for another.",
            "tp": "A true positive needs both: the label positive, and the classifier saying positive. With 0.423 it says negative. See [Logistic regression on words](/course/open/nlp-logistic-text?lang=en).",
            "fp": "A false positive is a sentence the classifier calls positive while its label is negative; here it is the other way round. See [Logistic regression on words](/course/open/nlp-logistic-text?lang=en).",
            "tn": "A true negative is a negative sentence called negative, and this sentence is labeled positive. See [Logistic regression on words](/course/open/nlp-logistic-text?lang=en).",
        },
    )
    quick_sentence
    return (quick_sentence,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    So far in this domain, a classifier chose between two labels, such as positive and negative. An assistant you talk to chooses among many: *wake me up at nine am on friday* asks for an alarm, *is it raining* for the weather. Each kind of request is an **intent**, with a name such as alarm_set or weather_query, and a classifier gives each command one intent.

    The commands of this lesson come from MASSIVE, a set that Jack FitzGerald and colleagues at Amazon published in 2022: "MASSIVE contains 1M realistic, parallel, labeled virtual assistant utterances spanning 51 languages, 18 domains, 60 intents, and 55 slots." A domain is a topic, such as music or the calendar; slots are the words of a command that matter, such as *five am*, and this lesson leaves them aside. Professional translators turned the English commands into 50 other languages, Dutch among them. This lesson uses the 16,521 English commands, each with one of the 60 intents: 11,514 train a classifier, 2,974 test it, and the other 2,033 are not used here. The intents are far from equally common. 810 training commands are calendar_set, which puts a reminder in the calendar, such as *remind me at one p. m.*; 14 are music_dislikeness, which says a song is not liked.

    The classifier is naive Bayes, from *Spam or not: naive Bayes*, with 60 intents instead of two labels. The score of an intent is the log of its share of the training commands, the prior, plus for each word of the command the log of the word's share of the words in the training commands of that intent, with the add-k smoothing of *Pairs you never saw: smoothing* and $k = 0.1$. Each test command gets the intent with the highest score, and 2,437 of the 2,974 get their own: 81.9%. But is the classifier equally good at every intent?

    **An everyday example.** A bus company says that 95% of all its trips run on time. Most trips are on three busy lines; on a quiet line with a few trips a day, half run late. The share of all trips hardly notices that line. The mean of the shares of the lines, with each line counting once, does.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Naive Bayes gives 81.9% of the 2,974 test commands their own intent. Work out the F1 score of each of the 59 intents of the test commands on its own, and take the plain mean of the 59 numbers. What do you expect?",
        options={
            "higher": "About 0.95: higher, as most intents are easy",
            "same": "About 0.82: the same as the share right",
            "right": "About 0.75: lower than the share right",
            "far": "About 0.45: far lower than the share right",
        },
        reveal="**About 0.75: lower**, 0.752. The intents with many commands mostly get a high F1 score: calendar_set 0.888, play_music 0.883, weather_query 0.909. But three rare intents get 0: general_greet, iot_hue_lighton and music_dislikeness, which naive Bayes gives to no command at all. Each of them counts as much in the mean as calendar_set, with its 209 test commands. The 30 intents with the most training commands have a mean F1 score of 0.825, the other 29 of 0.676.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### One label at a time

    Precision, recall and the F1 score of [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en) need a positive label. With many labels, take each label in turn as the positive one, and all the others together as negative. For play_music, a true positive is a command of play_music that the classifier gives play_music; a false positive is a command of another intent that it gives play_music; a false negative is a command of play_music that gets another intent. In the confusion matrix of *Beyond accuracy: precision and recall*, with a row for each true label and a column for each label the classifier gives, the true positives of a label are its cell on the diagonal, its false positives the rest of its column, and its false negatives the rest of its row.

    So one mistake counts twice. *do not play rock metal* is a command of music_dislikeness, and naive Bayes gives it play_music: a false positive for play_music, and a false negative for music_dislikeness.

    ### Two ways to average

    That gives an F1 score for each of the 59 intents. Daniel Jurafsky and James H. Martin describe the two ways to make one number of them in *Speech and Language Processing*: "In macroaveraging, we compute the performance for each class, and then average over classes. In microaveraging, we collect the decisions for all classes into a single confusion matrix, and then compute precision and recall from that table." A class is a label here. For the F1 score, with $L$ labels and the counts $TP_c$, $FP_c$ and $FN_c$ of label $c$:

    $$\text{macro } F_1 = \frac{1}{L} \sum_{c} \frac{2\,TP_c}{2\,TP_c + FP_c + FN_c} \qquad \text{micro } F_1 = \frac{2\,TP}{2\,TP + FP + FN}$$

    with $TP = \sum_c TP_c$, the true positives of all labels added up, and the same for $FP$ and $FN$. **Macro-averaging** gives every label the same weight, however few commands it has. **Micro-averaging** gives every command the same weight, so a common label weighs more. The picture shows both for naive Bayes on the test commands.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    Each gray bar is one intent, from the most training commands at the left to the fewest at the right. The solid line is the macro F1, 0.752; the dashed line the micro F1, 0.819. Jurafsky and Martin on the difference: "a microaverage is dominated by the more frequent class (in this case spam), since the counts are pooled. The macroaverage better reflects the statistics of the smaller classes, and so is more appropriate when performance on all the classes is equally important." Their example has three labels, urgent, normal and spam; the worked example below uses its numbers.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **intent** | what someone wants with a command, as a label, such as alarm_set |
        | **macro-averaging** | working out a score for each label and taking the plain mean: every label counts the same |
        | **micro-averaging** | adding up the counts of all labels and working out the score once: every example counts the same |
        | macro $F_1$, micro $F_1$ | the F1 score with macro-averaging and with micro-averaging |
        | $TP_c$, $FP_c$, $FN_c$ | the true positives, false positives and false negatives of label $c$ |
        """),
        L.watch_out("Some papers work out a macro F1 in another way: as the F1 score of the mean precision and the mean recall of the labels. For naive Bayes on the test commands, that gives 0.767 instead of 0.752. In this course, the macro F1 is the mean of the F1 scores. When you read or report a macro F1, check which of the two it is."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Fix the list of labels: here the 59 intents of the test commands.
    2. For each label, count its true positives, false positives and false negatives, with that label as positive and all the others as negative.
    3. For the macro F1, work out the F1 score of each label from its own counts, $2\,TP_c / (2\,TP_c + FP_c + FN_c)$, and take the plain mean over the labels.
    4. For the micro F1, add up the true positives, false positives and false negatives of all labels, and work out one F1 score from the three sums.
    5. Report both, and look at the labels with the lowest F1 score: that is where the classifier fails.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The example of Jurafsky and Martin sorts 367 mails into urgent, normal and spam. The course's `SLP_MATRIX` holds its confusion matrix, with a row for each true label and a column for each label the system gave, as in *Beyond accuracy: precision and recall*; Jurafsky and Martin draw it the other way round. For each label, the code takes the true positives from the diagonal, the false positives from the rest of its column, and the false negatives from the rest of its row.
    """)
    return


@app.cell
def example_cell():
    names = ["urgent", "normal", "spam"]
    cm = C.SLP_MATRIX   # a row for each true label, a column for each label the system gave
    for k in range(3):
        tp = cm[k][k]
        fp = cm[0][k] + cm[1][k] + cm[2][k] - tp
        fn = cm[k][0] + cm[k][1] + cm[k][2] - tp
        print(names[k], tp, fp, fn)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Urgent gets $F_1 = 2 \times 8 / (2 \times 8 + 11 + 8) = 16 / 35 = 0.457$, normal $120 / 215 = 0.558$ and spam $400 / 484 = 0.826$. The macro F1 is their mean, $(0.457 + 0.558 + 0.826) / 3 = 0.614$. For the micro F1, add up: $TP = 8 + 60 + 200 = 268$, $FP = 11 + 55 + 33 = 99$ and $FN = 8 + 40 + 51 = 99$, so the micro F1 is $536 / (536 + 99 + 99) = 0.730$. Spam has 251 of the 367 mails and pulls the micro F1 toward its own 0.826; urgent, with 16 mails, counts for a third in the macro F1. Jurafsky and Martin work out the precision this way: 0.60 with macro-averaging and 0.73 with micro-averaging.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_counts():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. counts: one label among many

        `gold` holds the true intent of each test command, and `said` the intent naive Bayes gives it, in the same order; the cell below prints the first three commands with both, from the course's `test`. Write `counts(gold, said, label)`: go over the places `i` of the commands, and count the true positives (both `gold[i]` and `said[i]` are `label`), the false positives (only `said[i]` is `label`) and the false negatives (only `gold[i]` is `label`). Give back `tp, fp, fn`.

        `small_gold` and `small_said`, the course's `SMALL_GOLD` and `SMALL_SAID`, are six commands: five of play_music and one of alarm_set, and a classifier that says play_music for all six. So `counts(small_gold, small_said, "play_music")` gives `(5, 1, 0)`, and `counts(small_gold, small_said, "alarm_set")` gives `(0, 0, 1)`. Replace the `...`. The check calls those two, then cases of its own: the labels of `more_gold` and `more_said`, the course's `MORE_GOLD` and `MORE_SAID`, a label that is in neither list, and three intents of the test commands, with `gold` and `said` from the course's `gold` and `said`.
        """),
        L.task("Count the true positives, false positives and false negatives of one label."),
    ])
    return


@app.cell
def commands_cell():
    small_gold = C.SMALL_GOLD
    small_said = C.SMALL_SAID
    more_gold = C.MORE_GOLD
    more_said = C.MORE_SAID
    gold = C.gold()
    said = C.said()
    print(small_gold)
    print(small_said)
    print(len(gold), len(said))
    for i in range(3):
        print(C.test()[i][1], "|", gold[i], "|", said[i])
    return gold, more_gold, more_said, said, small_gold, small_said


@app.cell(hide_code=True)
def exercise_counts():
    box_counts = L.code_box("counts", lines=12)
    box_counts
    return (box_counts,)


@app.cell(hide_code=True)
def check_counts(box_counts):
    result_counts = L.check("counts", box_counts.value)
    result_counts
    return (result_counts,)


@app.cell(hide_code=True)
def hints_counts():
    L.hints(
        "counts",
        [
            "Start `tp`, `fp` and `fn` at 0, and go over the places with `for i in range(len(gold)):`. Compare `gold[i]` and `said[i]` with `label`, and add 1 to the count that fits: `tp` when both are `label`, `fp` when only `said[i]` is, `fn` when only `gold[i]` is. After the loop, give back `tp, fp, fn`.",
            "In `small_gold`, the third command is alarm_set, and the classifier says play_music. For the label play_music, which count does that command go to, and what is true of its own intent?",
            r"`        if ___ and said[i] == label:`",
        ],
        why=r"`tp`, `fp` and `fn` start at 0; `for i in range(len(gold)):` with `if gold[i] == label and said[i] == label:` and `tp = tp + 1`, `if gold[i] != label and said[i] == label:` and `fp = fp + 1`, and `if gold[i] == label and said[i] != label:` and `fn = fn + 1`; after the loop, `return tp, fp, fn`.",
    )
    return


@app.cell(hide_code=True)
def text_after_counts():
    mo.md(r"""
    In the picture, your function counted for eight intents. calendar_set, the most common, gets 191 of its 209 test commands right, and 30 commands of other intents as well. general_quirky, for odd questions such as *check my car is ready*, has the most mistakes both ways: 54 false positives and 92 false negatives. Of the four rare intents in the picture, three never get a command: *what's up olly*, the one test command of general_greet, gets news_query, and *do not play rock metal* gets play_music.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_macro():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. macro_f1: every label counts the same

        Write `macro_f1(gold, said, labels)`: for each label of `labels`, get its counts from `counts`, work out its F1 score, $2\,TP / (2\,TP + FP + FN)$, and give back the plain mean over the labels. Every label in `labels` occurs in `gold`, so no denominator is 0. The course's `counts` is there for you to use.

        `small_labels`, the course's `SMALL_LABELS`, holds alarm_set and play_music. For the small example, play_music gets $2 \times 5 / (2 \times 5 + 1 + 0) = 10 / 11 = 0.909$ and alarm_set $0 / (0 + 0 + 1) = 0$, so `macro_f1(small_gold, small_said, small_labels)` gives $(0.909 + 0) / 2 = 0.4545$. Replace the `...`. The check calls that and `macro_f1(more_gold, more_said, more_labels)`, with `more_labels` the course's `MORE_LABELS`, then cases of its own: every command right, the second example with only alarm_set and weather_query as labels, and the test commands with `labels`, the 59 intents from the course's `labels`: with naive Bayes, without its prior from the course's `said_flat`, and knowing only 10 intents from the course's `said_known`.
        """),
        L.task("Work out the macro F1: the mean of the F1 scores of the labels."),
    ])
    return


@app.cell
def labels_cell():
    small_labels = C.SMALL_LABELS
    more_labels = C.MORE_LABELS
    labels = C.labels()
    print(small_labels, more_labels)
    print(labels[:4], labels[-1])
    return labels, more_labels, small_labels


@app.cell(hide_code=True)
def exercise_macro():
    box_macro = L.code_box("macro", lines=6)
    box_macro
    return (box_macro,)


@app.cell(hide_code=True)
def check_macro(box_macro):
    result_macro = L.check("macro", box_macro.value)
    result_macro
    return (result_macro,)


@app.cell(hide_code=True)
def hints_macro():
    L.hints(
        "macro",
        [
            "Start with `total = 0`, and go over the labels with `for label in labels:`. For each, get `tp, fp, fn` from `counts(gold, said, label)`, and add its F1 score to `total`. After the loop, turn `total` into the plain mean over the labels.",
            "For the small example the two F1 scores are 0.909 and 0, and the macro F1 is 0.4545. What do you divide their sum by: the 6 commands, or something else?",
            r"`    return total / ___`",
        ],
        why=r"`total = 0`; `for label in labels:` with `tp, fp, fn = counts(gold, said, label)` and `total = total + 2 * tp / (2 * tp + fp + fn)`; after the loop, `return total / len(labels)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_macro():
    mo.md(r"""
    In the picture, your function took the mean of the 59 bars: 0.752. Three bars are 0, and music_settings and audio_volume_other get 0.286: each finds 1 of its 6 test commands. Without the three intents that get 0, the mean of the other 56 would be 0.792.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_micro():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. micro_f1: every command counts the same

        Write `micro_f1(gold, said, labels)`: add up the true positives, false positives and false negatives of all labels in `labels`, which `counts` gives, in three sums, and give back one F1 score of the three sums.

        For the small example, the sums are $TP = 0 + 5 = 5$, $FP = 0 + 1 = 1$ and $FN = 1 + 0 = 1$, so `micro_f1(small_gold, small_said, small_labels)` gives $10 / (10 + 1 + 1) = 0.8333$: almost twice the macro F1. Replace the `...`. The check calls that and `micro_f1(more_gold, more_said, more_labels)`, then the same cases of its own as for `macro_f1`.
        """),
        L.task("Work out the micro F1: one F1 score of the counts of all labels added up."),
    ])
    return


@app.cell(hide_code=True)
def exercise_micro():
    box_micro = L.code_box("micro", lines=10)
    box_micro
    return (box_micro,)


@app.cell(hide_code=True)
def check_micro(box_micro):
    result_micro = L.check("micro", box_micro.value)
    result_micro
    return (result_micro,)


@app.cell(hide_code=True)
def hints_micro():
    L.hints(
        "micro",
        [
            "Start `all_tp`, `all_fp` and `all_fn` at 0, and go over the labels with `for label in labels:`. For each, get `tp, fp, fn` from `counts(gold, said, label)`, and add each of the three to its sum. After the loop, give back the F1 score of the three sums.",
            "For the small example the sums are 5, 1 and 1. What is the F1 score of those counts, and do you work it out inside the loop or after it?",
            r"`        all_fp = all_fp + ___`",
        ],
        why=r"`all_tp`, `all_fp` and `all_fn` start at 0; `for label in labels:` with `tp, fp, fn = counts(gold, said, label)`, then `all_tp = all_tp + tp`, `all_fp = all_fp + fp` and `all_fn = all_fn + fn`; after the loop, `return 2 * all_tp / (2 * all_tp + all_fp + all_fn)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_micro():
    mo.md(r"""
    In the picture, your micro F1 is 0.8194: the same number as the share of the test commands that get their own intent, 2,437 of 2,974. That is not a coincidence. A command that gets its own intent is one true positive, for that intent. A command that gets another intent is one false positive, for the intent it gets, and one false negative, for its own. So $TP$ is the number of commands right, 2,437, and $FP$ and $FN$ are both the number wrong, 537:

    $$\text{micro } F_1 = \frac{2 \times 2437}{2 \times 2437 + 537 + 537} = \frac{2437}{2437 + 537} = 0.819$$

    When each example has exactly one label, and the classifier only gives labels of the list, the micro precision, the micro recall and the micro F1 are all the accuracy. For a table that holds every label the classifier gives, as in their example, Jurafsky and Martin put it in a footnote: "Note that the microaverage precision is always equal to the microaverage recall." Without its prior, naive Bayes gives three test commands, such as *baseball scores*, cooking_query, an intent with 4 training commands and no test commands. Those three mistakes are false negatives for their own intent, but false positives for none of the 59 labels. Over the 59 intents, its micro precision is then 2,434 of 2,971 and its micro recall 2,434 of 2,974, so its micro F1, 0.8188, is a little higher than its accuracy, 0.8184.

    ### How good is it?

    The makers of MASSIVE report, for each language, the share of test commands that get their own intent, which they call "intent accuracy", and for the slots a "micro-averaged slot F1 score". Three far larger models, with 258 to 580 million parameters and trained on the training commands of all 51 languages together, get 87.9% to 89.0% of the English test commands right; naive Bayes, which counts words, gets 81.9%. On the Dutch commands, the same models get 86.8% to 87.6%, and naive Bayes gets 79.9%, with a macro F1 of 0.730: there four intents get 0.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: fewer intents

    What if the classifier knew only the most common intents? The slider sets how many of the 60 intents naive Bayes knows, counted from the one with the most training commands. It learns each of them from the same training commands as before, and gives every test command the known intent with the highest score, so the commands of an intent it does not know always go wrong. The picture shows the F1 score of each intent, with the unknown ones in the purple area, and the two averages, worked out by your `micro_f1` and `macro_f1` once exercises 2 and 3 are right, and by the course's until then.
    """)
    return


@app.cell(hide_code=True)
def demo_cell(box_macro, box_micro, result_macro, result_micro):
    _names = C.EXERCISES["macro"].names
    _macro = C.mine(box_macro.value, result_macro.state == "correct", "macro_f1", _names)
    _micro = C.mine(box_micro.value, result_micro.state == "correct", "micro_f1", _names)
    _values = C.demo_values(_macro, _micro)
    demo_known = L.explore("known", "intents known", lambda _k: C.demo_frame(_k, _values, L.lang),
                           start=60, values=C.KNOWN, role="",
                           task="Slide to 30 intents, about half of them. How much does each average lose?")
    demo_known
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With 30 intents, naive Bayes still knows the intent of 2,382 of the 2,974 test commands, 80%, and the micro F1 drops from 0.819 to 0.677. The macro F1 falls to 0.380: 29 of the 59 intents now get 0. With 50 intents, only 74 test commands belong to an intent it does not know: the micro F1 goes from 0.819 to 0.806, the macro F1 from 0.752 to 0.682. With 1, every command gets calendar_set: a micro F1 of 0.070, the 209 commands of calendar_set out of 2,974, and a macro F1 of 0.002. A classifier that leaves out the rare labels can look almost as good on the micro F1 as one that does not; the macro F1 shows the difference.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_prior():
    mo.md(r"""
    ### Exercise 4. With or without the prior?
    """)
    return


@app.cell(hide_code=True)
def choice_prior_cell():
    choice_prior = L.choice(
        "prior",
        question="The score of naive Bayes starts with the log of the share of the training commands of each intent: the prior. Without it, as if every intent were equally common, it gives the 29 intents of the test commands with the fewest training commands to 635 test commands instead of 534; those 29 are the true intent of 592. It then gets 2,434 test commands right instead of 2,437, and a macro F1 of 0.763 instead of 0.752. Which of the two is better?",
        options={
            "prior": "With the prior, since more commands come out right",
            "flat": "Without, since its macro F1 is higher",
            "right": "That depends on which average the task needs",
            "same": "They are the same, since their micro F1 is 0.819 for both",
        },
        correct="right",
        why={
            "right": "The micro F1 counts every command: with the prior, 3 more of the 2,974 come out right, and the micro F1 is 0.8194 against 0.8188. The macro F1 counts every intent: without the prior, the rare intents are given more often, music_dislikeness goes from 0 to 0.571 and general_greet from 0 to 0.333. Even calendar_set gets a higher F1 score, 0.903 instead of 0.888, as fewer commands of other intents get it. The 29 rare intents are the true intent of 592 commands: with the prior they are given too rarely, 534 times, and without it a little too often, 635 times. Which version is better depends on what counts for the task: every command, or every intent.",
            "prior": "More commands right is what the micro F1 counts, and the difference is 3 of 2,974. The macro F1 says the opposite. Which average counts depends on the task. See *The idea*.",
            "flat": "A higher macro F1 means that the intents do better on average, but 3 more commands go wrong. Which average counts depends on the task. See *The idea*.",
            "same": "Their micro F1 is about the same, but their macro F1 differs by 0.012: without the prior, the rare intents do better. See *The idea*.",
        },
    )
    choice_prior
    return (choice_prior,)


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Exercise 5. Which average?

    Each situation needs one number for a classifier with many labels. Put each under the average that fits it best.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_average = L.sort("which-average")
    sort_average
    return (sort_average,)


@app.cell(hide_code=True)
def check_sort(sort_average):
    result_sort = L.check("which-average", sort_average)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en) gave one F1 score for one positive label; this lesson took every label in turn as the positive one, on the intents of MASSIVE, and made one number of the 59 F1 scores in two ways. Macro-averaging gives every label the same weight, so the rare intents that naive Bayes misses pull it down; micro-averaging gives every command the same weight, and with one intent for each command it is the accuracy. The same two averages measure any classifier with many labels, whatever it does inside. The next lesson of the domain, *Project: a spam filter with TF-IDF*, builds a whole classifier for real text messages.

    **An everyday example.** A school report gives a grade for each subject. The mean of the grades is like the macro F1: a small subject counts as much as a big one. Counting all right answers of all tests together is like the micro F1: a subject with many tests counts more.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_prior, result_counts, result_macro, result_micro, result_sort):
    L.wrap_up(result_counts, result_macro, result_micro, choice_prior, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
