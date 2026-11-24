# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-project-sentiment. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("nlp-project-sentiment", "en")
    C = L.checks   # the review sentences in English and Dutch, the course's classifier and lexicon, the demos and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [Many classes: micro and macro averages](/course/open/nlp-micro-macro?lang=en), and one about [A list of good and bad words](/course/open/nlp-sentiment-lexicon?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_macro():
    quick_macro = L.choice(
        "quick-macro",
        question="A classifier calls all 10 sentences of a test positive, and 6 of them are positive. What is its macro F1 over the two labels, positive and negative?",
        options={
            "accuracy": "0.6",
            "positive": "0.75",
            "right": "0.375",
            "none": "It cannot be worked out, since no sentence is called negative",
        },
        correct="right",
        why={
            "right": r"For positive: $2 \times 6 / (2 \times 6 + 4 + 0) = 0.75$. For negative: no true positives, no false positives and 4 false negatives, so $0 / (0 + 0 + 4) = 0$. The mean is $0.375$. In this project, a sentence can also get no label at all, and your `macro_f1` counts that too.",
            "accuracy": "0.6 is how many of the 10 it gets right. Macro F1 works out the F1 score of each label, and takes their mean. What is the F1 score of negative, which it never gives? See [Many classes: micro and macro averages](/course/open/nlp-micro-macro?lang=en).",
            "positive": r"0.75 is the F1 score of positive alone, $2 \times 6 / (2 \times 6 + 4 + 0)$. Macro F1 takes the mean over both labels. What does negative add? See [Many classes: micro and macro averages](/course/open/nlp-micro-macro?lang=en).",
            "none": r"The F1 score of negative has 4 false negatives under its fraction: $0 / (0 + 0 + 4)$, which is a number. What is it, and what is the mean with the F1 score of positive? See [Many classes: micro and macro averages](/course/open/nlp-micro-macro?lang=en).",
        },
    )
    quick_macro
    return (quick_macro,)


@app.cell(hide_code=True)
def quick_check_negation():
    quick_negation = L.choice(
        "quick-negation",
        question="In a lexicon, *not* has the score −0.6 and *bad* −0.7. A negation turns the scores of the next two words around, and keeps its own. Which sum does *not bad* get?",
        options={
            "plain": "−1.3",
            "right": "0.1",
            "only": "0.7",
            "both": "1.3",
        },
        correct="right",
        why={
            "right": "*not* adds its own −0.6, and *bad*, right after it, counts the other way around: +0.7. The sum is 0.1, so *not bad* comes out positive. In this project, `lexicon_labels` makes a label from such sums.",
            "plain": "−1.3 adds both scores as they are, without the negation. What happens to the score of *bad*, right after *not*? See [A list of good and bad words](/course/open/nlp-sentiment-lexicon?lang=en).",
            "only": "0.7 turns *bad* around, but leaves out the score of *not* itself. Does a negation keep its own score? See [A list of good and bad words](/course/open/nlp-sentiment-lexicon?lang=en).",
            "both": "1.3 turns *not* around as well. The negation turns the words after it around; what happens to its own score? See [A list of good and bad words](/course/open/nlp-sentiment-lexicon?lang=en).",
        },
    )
    quick_negation
    return (quick_negation,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        In [Project: a spam filter with TF-IDF](/course/open/nlp-project-spam?lang=en) you built a text classifier of your own, piece by piece: `features` for the TF-IDF rows and `step` for the training, both in your project `text`. In [A list of good and bad words](/course/open/nlp-sentiment-lexicon?lang=en) you built a sentiment lexicon from counts. Two ways to tell a good review from a bad one: one learns all its weights together, the other gives each word a score from its own two counts. In this step of the project you run both on the same reviews, in English and in Dutch, and you measure them the same way.

        One number for each is the start. Macro F1, from [Many classes: micro and macro averages](/course/open/nlp-micro-macro?lang=en), counts both labels equally, and a lexicon can leave a sentence without a label, so your project counts that too. Then the mistakes themselves: if two classifiers that work so differently go wrong on different sentences, each covers for the other. If they go wrong on the same sentences, the trouble lies in those sentences, and a better method of the same kind will not get far.

        **An everyday example.** Two teachers mark the same essays, one with a rubric, the other by reading the whole essay. Where they disagree, a second look helps. Where both give a low mark to an essay that the student is sure is good, the cause is probably in the essay, or in the assignment. That is only a comparison; below you count where your two classifiers agree.
        """),
        draw.project_steps(["Your spam filter", "Your sentiment classifier"], 1,
                           here_label="← today", title="Where your project stands: text"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_shared = L.guess(
        "shared",
        question="The classifier of *Logistic regression on words* gets 109 of the 600 English held-out sentences wrong, and the lexicon of *A list of good and bad words*, with negation, gets 106 wrong. In how many of the 600 sentences are both wrong, do you think?",
        options={
            "chance": "About 20, as if their mistakes had nothing to do with each other",
            "some": "About 45",
            "right": "About 75",
            "all": "About 105: nearly every mistake",
        },
        reveal="**74.** If the mistakes of the two had nothing to do with each other, about $109 \\times 106 / 600 \\approx 19$ sentences would be wrong in both. 74 is almost four times as many. Only 35 sentences are wrong in the classifier alone and 32 in the lexicon alone. In Dutch, it is 89 in both, 31 and 30. Exercise 3 finds these sentences, and the end of the lesson reads them.",
    )
    guess_shared
    return (guess_shared,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **The sentences.** The same review sentences as in [Logistic regression on words](/course/open/nlp-logistic-text?lang=en) and [A list of good and bad words](/course/open/nlp-sentiment-lexicon?lang=en): 3,000 sentences from Amazon, IMDb and Yelp, collected by Dimitrios Kotzias, Misha Denil, Nando de Freitas and Padhraic Smyth in 2015, each labeled positive or negative, and the course's Dutch translation of all 3,000. Every fifth sentence is held out, as before. In each language, that gives 2,400 training sentences and 600 held-out sentences, 291 of them positive.

    **Your classifier.** The spam filter of your project, on other text: your `document_frequencies` and `features` make a TF-IDF row of each sentence, and your `step` trains a weight for each word of the training sentences, 4,518 words in English and 4,737 in Dutch, in 300 steps with learning rate 1.0, from weight 0 and bias 0. A held-out sentence is positive when its probability of positive, $p$, is 0.5 or more. The code is the same in both languages; only the words differ.

    **Your lexicon.** The list of [A list of good and bad words](/course/open/nlp-sentiment-lexicon?lang=en): each word of the training sentences gets a score from its two counts, $\log_{10}\frac{\text{positive} + 1}{\text{negative} + 1}$, and a sentence gets the sum of the scores of its words, with the two words after a negation turned around. Above 0 is positive, below 0 negative, and at exactly 0 the sentence gets **no label**: the lexicon says nothing about it. Pattern's list comes along as a third, made partly by hand and without these reviews.

    **Macro F1 with no label.** [Many classes: micro and macro averages](/course/open/nlp-micro-macro?lang=en) worked out the F1 score of each label, $2\,TP / (2\,TP + FP + FN)$, and took their mean. Here there are two labels, positive and negative, and a sentence may get neither. It then counts as a false negative of its own label, and as a false positive of no label. So a sentence without a label costs recall, and leaves precision alone.

    **The mistakes, side by side.** One number for each classifier hides which sentences each gets wrong. So your project sorts the held-out sentences into three groups: those both get wrong, those only your classifier gets wrong, and those only your lexicon gets wrong.

    The picture follows one held-out sentence through both, with the weights after training, as you will train them below.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.vstack([
        mo.md(r"""
        The classifier adds a value times a weight for each word. *not* and *bad* both have a strongly negative weight, −1.781 and −1.493, since both stand mostly in complaints. So $z = -4.624$ and $p = 0.010$: negative, and wrong. The lexicon turns the scores of *a* and *bad*, the two words after *not*, around: *bad* now counts +1.45, and the sum is +0.27: positive, the sentence's own label. A TF-IDF row holds no order, so the classifier cannot see that *not* belongs to *bad*.

        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **no label** | what a lexicon gives a sentence whose sum is exactly 0; in code `None` |
        | `C.training(lang)` | the 2,400 training sentences in English (`"en"`) or Dutch (`"nl"`), as a list of `[words, label]`: the words in small letters first, then the label, 1 for positive and 0 for negative |
        | `C.held_out(lang)` | the 600 held-out sentences, in the same form |
        """),
        L.watch_out("The two classifiers lie close: 491 against 494 of the 600 English held-out sentences right. With 600 sentences, a difference of a few sentences says little about which one is better: another 600 reviews could turn it around. A difference you can trust needs many more sentences, or the same difference again on other data."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. In each language, hold out every fifth sentence: 2,400 to train on, 600 held out.
    2. Your classifier: TF-IDF rows with your `features`, 300 steps of your `step` with learning rate 1.0, and positive from $p = 0.5$ on.
    3. Your lexicon: a score for each word from its counts in the training sentences, and a label for each sentence from its sum, with negation. That is `lexicon_labels`.
    4. The macro F1 of each, in each language. That is `macro_f1`.
    5. The sentences both get wrong, and those each gets wrong alone. That is `compare`. Then read them.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell below follows step 1, for the lesson's language. It prints how many training and held-out sentences there are, and the second training sentence, to show their form. Then it counts the positive held-out sentences, by adding up their labels. Run the cell.
    """)
    return


@app.cell
def example_cell():
    # the training and the held-out sentences: pairs of [words, label], label 1 for positive
    _training = C.training(L.lang)
    _held = C.held_out(L.lang)
    print(len(_training), "training sentences,", len(_held), "held out")
    print(_training[1])
    _positive = 0
    for _words, _label in _held:
        _positive = _positive + _label
    print(_positive, "held-out sentences are positive")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Each sentence is a pair: first the list of its words, then its label. That is the form of the messages of your spam filter, so your `features` and `step` take these sentences as they are. *A list of good and bad words* put the label first; in this project, the words come first everywhere, also in the course's `lexicon` in exercise 1. 291 of the 600 held-out sentences are positive and 309 negative: both labels are about as frequent, so macro F1 and how often a classifier is right lie close here. They part where a classifier leaves sentences without a label, or says one label much more often than the other.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lexicon_labels():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. lexicon_labels: the labels of your lexicon

        `lexicon_labels(texts, scores, negations)` gets a list `texts` with the words of each sentence, a lexicon `scores` with a score for each word, and the list `negations`. It gives back a list with one label for each sentence: 1 when the sum of its scores is above 0, 0 when it is below 0, and `None` when it is exactly 0. The box already has the course's versions of your `lexicon` and your `score` from *A list of good and bad words*, with two changes: `lexicon` takes each sentence as `[words, label]`, and `score` calls the lexicon `scores`, so that the name `lexicon` stays free for the function.

        `small_texts`, the course's `SMALL_TEXTS`, holds four short sentences: *good food*, *not good*, *not bad* and *fine*. `small_scores`, the course's `SMALL_SCORES`, gives *good* 0.5, *bad* −0.7 and *not* −0.6. *good food* sums to 0.5: positive. *not good* sums to −0.6 − 0.5 = −1.1: negative. *not bad* sums to −0.6 + 0.7 = 0.1: positive. The lexicon does not have *fine*, so that sum is 0: no label. `lexicon_labels(small_texts, small_scores, ["not"])` gives `[1, 0, 1, None]`. Replace the `...`. The check calls that, then cases of its own: no sentences at all; *not fine bad*, where *bad* is the second word after *not*; *bad not*, with the negation as the last word; two words whose scores add up to exactly 0; two Dutch sentences with *niet* and *geen*; and the first five English held-out sentences, with the course's lexicon from the training sentences. When it is right, the check saves the box in your project, in `work/projects/text/lexicon.py`.
        """),
        L.task("Give each sentence a label from the sum of its scores, and no label for a sum of exactly 0."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lexicon_labels():
    box_lexicon_labels = L.code_box("lexicon_labels", lines=24)
    box_lexicon_labels
    return (box_lexicon_labels,)


@app.cell(hide_code=True)
def check_lexicon_labels(box_lexicon_labels):
    result_lexicon_labels = L.check("lexicon_labels", box_lexicon_labels.value)
    result_lexicon_labels
    return (result_lexicon_labels,)


@app.cell(hide_code=True)
def hints_lexicon_labels():
    L.hints(
        "lexicon_labels",
        [
            "In the loop, `total` is the sum of the sentence, from the course's `score`. Then `label`: give it a value first, and change it with one `if` for a sum above 0 and one for a sum below 0. Last, add `label` to `labels` with `append`.",
            "A sum of exactly 0 is neither above 0 nor below 0, so neither `if` changes the label. Which value should the label have before the two `if` lines, then?",
            "`        label = ___`",
        ],
        why="`total = score(words, scores, negations)`, then `label = None`, `if total > 0:` with `label = 1` under it, `if total < 0:` with `label = 0`, and `labels.append(label)`; after the loop, `return labels`. The first five English held-out sentences get `[1, 0, 0, 0, 0]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_lexicon_labels():
    mo.md(r"""
    The lexicon from the training sentences gives every English held-out sentence a label, 305 positive and 295 negative, and leaves one Dutch sentence without one, *Magische hulp.*: neither *magische* nor *hulp* is in any training sentence, so the sum is 0. Pattern's list leaves 131 English sentences without a label, and 175 Dutch ones. In 98 of those English sentences, not one word is on its list. Of the 131, 88 are negative: Pattern's list knows mostly adjectives, and complaints often do without them, as in *Doesn't hold charge.*
    """)
    return


@app.cell(hide_code=True)
def text_exercise_macro_f1():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. macro_f1: macro F1 with no label

        `macro_f1(gold, said)` gets the labels of the sentences, `gold`, and the labels a classifier gives them, `said`: 1, 0, or `None` for no label. It gives back the macro F1 over the labels 1 and 0: the mean of their F1 scores, each $2\,TP / (2\,TP + FP + FN)$, as in [Many classes: micro and macro averages](/course/open/nlp-micro-macro?lang=en). A `None` is a label for neither: it is a false negative of the sentence's own label, and a false positive of no label. Both labels occur in `gold`. The box already has the loop over the two labels and the last line.

        `small_gold`, the course's `SMALL_GOLD`, holds three positive and then three negative sentences, and `small_said`, the course's `SMALL_SAID`, gives them `[1, 1, 0, 0, None, 1]`. For label 1: two true positives, one false positive (the sixth) and one false negative (the third), so $F_1 = 4 / (4 + 1 + 1) = 0.667$. For label 0: one true positive (the fourth), one false positive (the third) and two false negatives (the fifth, without a label, and the sixth), so $F_1 = 2 / (2 + 1 + 2) = 0.4$. `macro_f1(small_gold, small_said)` gives `0.533`. Replace the `...`. The check calls that, then cases of its own: four sentences that all get their own label; five sentences that all get 1, two of them positive; two sentences without a label; three sentences that all get the other label; the labels of the course's classifier on the 600 English held-out sentences; and those of Pattern's list on the 600 Dutch ones. It saves your function in `work/projects/text/scores.py`.
        """),
        L.task("Work out the F1 score of each label, with no label as a false negative, and give back their mean."),
    ])
    return


@app.cell(hide_code=True)
def exercise_macro_f1():
    box_macro_f1 = L.code_box("macro_f1", lines=18)
    box_macro_f1
    return (box_macro_f1,)


@app.cell(hide_code=True)
def check_macro_f1(box_macro_f1):
    result_macro_f1 = L.check("macro_f1", box_macro_f1.value)
    result_macro_f1
    return (result_macro_f1,)


@app.cell(hide_code=True)
def hints_macro_f1():
    L.hints(
        "macro_f1",
        [
            "For each label: `tp`, `fp` and `fn`, 0 at the start, and a loop `for i in range(len(gold)):` with three `if` lines that compare `gold[i]` and `said[i]` with `label`. After that loop, still inside the loop over the labels, add the F1 score of the label to `total`.",
            "A true positive of a label has that label in both lists, a false positive only in `said`, a false negative only in `gold`. A `None` in `said` is never equal to a label. How do the three counts make the F1 score?",
            "`        total = total + ___`",
        ],
        why="Under the loop over the places: `if gold[i] == label and said[i] == label:` with `tp = tp + 1`, `if gold[i] != label and said[i] == label:` with `fp = fp + 1`, and `if gold[i] == label and said[i] != label:` with `fn = fn + 1`; after that loop, `total = total + 2 * tp / (2 * tp + fp + fn)`. With the labels of the course's classifier on the English held-out sentences, it gives 0.818.",
    )
    return


@app.cell(hide_code=True)
def text_after_macro_f1():
    mo.md(r"""
    In English, the classifier gets 0.818 and the lexicon from the training sentences 0.823; in Dutch 0.800 and 0.802. Pattern's list gets 0.703 in English and 0.637 in Dutch. Its F1 score for negative is the lowest of all, 0.626 in English and 0.568 in Dutch: Pattern's list calls only 161 English sentences negative, against 308 positive, and leaves 131 without a label, 88 of them negative. The two classifiers built from these reviews are about equally good in both languages, and both do about two points worse in Dutch.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Train your classifier

    The button trains your classifier on the 2,400 training sentences, first in English, then in Dutch. Your `document_frequencies` and `features` make the rows, and your `step` takes 300 steps with learning rate 1.0, from weight 0 for every word and bias 0. A file that is not in your project, or does not pass, is replaced by the course's version, as the note under *Why this matters* says. The chart shows the mean log loss of the training sentences after each step. Beside it you see the five words with the strongest weight each way, three held-out sentences with their probability of positive, and how many held-out sentences get their own label at 0.5. It takes about ten seconds. Afterwards you can go back through the steps of the run in English.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_button = mo.ui.run_button(label="Train in English and in Dutch, 300 steps each")
    demo_button
    return (demo_button,)


@app.cell(hide_code=True)
def demo_run(demo_button):
    trained = None
    if not demo_button.value:
        _out = mo.md("*Press the button to train.*")
    else:
        _now = {}

        def _progress(_lang, _step, _loss, _ws, _b, _held_rows):
            if _lang not in _now:
                _words = C.live_words(L.lang, _lang)
                _now[_lang] = L.live({"title": _words["title"], "x": _words["x"], "y": _words["y"], "alt": _words["alt"]},
                                     C.STEPS, low=0.0, high=0.7)
                _now[_lang].above = [mo.Html(_now["en"].picture())] if _lang == "nl" and "en" in _now else []
                _now["words"] = _words
            _now[_lang].watch(lambda: C.now_parts(_held_rows, _ws, _b, L.lang, _lang))
            _now[_lang].add(_step, _loss, _now["words"]["progress"].format(s=_step, n=C.STEPS, loss=L.number(_loss, 3)))

        mo.output.replace(mo.md("*Making the rows of the sentences …*"))
        _run = C.demo(_progress)
        if isinstance(_run, str):
            _note = C.broken(_run, L.lang)
            _last = _now.get("nl", _now.get("en"))
            _out = _last.stopped(_note) if _last is not None else mo.md(_note)
        else:
            trained = _run
            _mine = _run[L.lang]
            _who = {(): "the course's functions", ("features",): "your `features` and the course's `step`",
                    ("train",): "the course's `features` and your `step`"}.get(tuple(_run["yours"]), "your `features` and your `step`")
            _out = mo.vstack([
                mo.md(f"""After {C.STEPS} steps with {_who}, your classifier gives <span class="ea-num">{C.rewind_right(_run["en"], C.STEPS)}</span> of the 600 English held-out sentences their own label at 0.5, and <span class="ea-num">{C.rewind_right(_run["nl"], C.STEPS)}</span> of the 600 Dutch ones. The mean log loss of the training sentences is {L.number(_run["en"]["curve"][-1][1], 3)} in English and {L.number(_run["nl"]["curve"][-1][1], 3)} in Dutch."""),
                L.explore("rewind", "step", lambda _s: (C.rewind_frame(_mine, int(_s), L.lang),
                                                        "After step {s}: {r} of the 600 English held-out sentences get their own label.".format(
                                                            s=int(_s), r=C.rewind_right(_mine, int(_s)))),
                          start=C.STEPS, values=C.LOOKS, role="",
                          task="Go back to step 1, then forward again: which words are strong first, and how far does the probability of positive of *Predictable, but not a bad watch.* fall?"),
            ])
    _out
    return (trained,)


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With every weight 0, every sentence gets $p = 0.5$ and the mean log loss is $\log 2 = 0.693$; after 300 steps it is 0.165 in English and 0.158 in Dutch. At 0.5, your classifier gives 491 of the 600 English held-out sentences their own label, and 480 of the Dutch ones. The strongest words toward positive are *great* (2.21), *good* (1.33), *excellent* (1.16), *nice* (1.14) and *love* (1.10); toward negative *not* (−1.78), *bad* (−1.49), *worst* (−1.06), *poor* (−0.99) and *terrible* (−0.85). In Dutch, *geweldig* (1.86) and *geweldige* (1.74) stand at the top, and *niet* (−2.17) and *slecht* (−1.47) at the bottom. Dutch adds an *e* to an adjective before a noun, so *geweldig* and *geweldige* each get a weight of their own. The Dutch training sentences have more different words, 4,737 against 4,518, and more words that occur only once, 2,922 against 2,625; whether that explains the 11 sentences between the two languages, these numbers alone do not show.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_compare():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. compare: where both go wrong

        `compare(gold, mine, theirs)` gets the labels of the sentences, `gold`, and the labels two classifiers give them, `mine` and `theirs`. It gives back three lists of places: the sentences that both get wrong, those that only `mine` gets wrong, and those that only `theirs` gets wrong. A label is wrong when it is not the sentence's own label, so a missing label, `None`, is wrong too. The box already has the three empty lists, the loop over the places and the last line.

        `small_mine`, the course's `SMALL_MINE`, is `[1, 0, 1, 1, 0, 0]`, and `small_theirs`, the course's `SMALL_THEIRS`, is `[1, 0, 0, 0, None, 0]`, both for `small_gold`, which is `[1, 1, 1, 0, 0, 0]`. Counting the places from 0, as Python does, the sentence at place 1 is wrong in both; the one at place 3 only in `mine`; those at places 2 and 4 only in `theirs`, which gives the one at place 4 no label. `compare(small_gold, small_mine, small_theirs)` gives `([1], [3], [2, 4])`. Replace the `...`. The check calls that, then cases of its own: three empty lists; three sentences that both get right; two sentences that both get wrong, one of them without a label in `theirs`; three sentences, one wrong only in `mine` and two only in `theirs`; and the course's classifier against the course's lexicon on the 600 English held-out sentences. It saves your function in `work/projects/text/compare.py`.
        """),
        L.task("Sort the places into wrong in both, wrong only in `mine` and wrong only in `theirs`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_compare():
    box_compare = L.code_box("compare", lines=16)
    box_compare
    return (box_compare,)


@app.cell(hide_code=True)
def check_compare(box_compare):
    result_compare = L.check("compare", box_compare.value)
    result_compare
    return (result_compare,)


@app.cell(hide_code=True)
def hints_compare():
    L.hints(
        "compare",
        [
            "Three `if` lines in the loop, one for each list. Each compares `mine[i]` and `theirs[i]` with `gold[i]`, joins the two comparisons with `and`, and adds the place `i` to its list with `append`.",
            "A sentence that only `theirs` gets wrong is right in `mine`, and in `theirs` it has another label than in `gold`, or none at all. Which comparison is true for two labels that differ?",
            "`        if mine[i] == gold[i] and theirs[i] ___ gold[i]:`",
        ],
        why="`if mine[i] != gold[i] and theirs[i] != gold[i]:` with `both.append(i)`, `if mine[i] != gold[i] and theirs[i] == gold[i]:` with `only_mine.append(i)`, and `if mine[i] == gold[i] and theirs[i] != gold[i]:` with `only_theirs.append(i)`. For the course's classifier and the course's lexicon on the English held-out sentences, the three lists hold 74, 35 and 32 places.",
    )
    return


@app.cell(hide_code=True)
def text_project_demo():
    mo.md(r"""
    ### What your project can do now

    Your project now has a classifier and a lexicon, and the functions to measure them and to compare their mistakes. The picture takes the classifier you trained above, your `lexicon_labels` with the course's lexicon from the training sentences, your `macro_f1` and your `compare`. It gives the macro F1 of both in both languages, and for the lesson's language a few sentences of each group: first those the text names, then the shortest. A file that is not in your project yet counts with the course's version, and the picture says so.
    """)
    return


@app.cell(hide_code=True)
def project_demo(result_compare, result_lexicon_labels, result_macro_f1, trained):
    _r = (result_lexicon_labels, result_macro_f1, result_compare)   # so that this cell runs again after each check
    if trained is None:
        _out = mo.md("*Train your classifier above first: the picture compares it with your lexicon.*")
    else:
        _out = mo.Html(C.project_picture(trained, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_after_project_demo():
    mo.md(r"""
    Read the three groups. Wrong in both: *Excellent starter wireless headset.* has the label negative, though nothing in it complains; perhaps whoever labeled it slipped, perhaps the review went on in another sentence. A label you cannot explain from the sentence itself defeats every classifier that reads only the sentence. Short sentences such as *Don't miss it.* go wrong in both as well: *don't* leans negative in these reviews, and here it is advice to go. Wrong only in your classifier: *Predictable, but not a bad watch.*, the sentence of the picture, and *This film has no redeeming features.*, where the lexicon turns *features* around after *no*. Wrong only in your lexicon: *Again, no plot at all.* *plot* already has a negative score of its own, −0.78: in these reviews, a plot is mostly mentioned to complain about it, 23 of its 26 times in negative sentences. Turning it around after *no* makes it count as praise. The rule that fixes one sentence breaks another, as in *A list of good and bad words*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shared():
    mo.md(r"""
    ### Exercise 4. What do the shared mistakes say?
    """)
    return


@app.cell(hide_code=True)
def choice_shared_cell():
    choice_shared = L.choice(
        "shared-mistakes",
        question="Your classifier and your lexicon share 74 of their mistakes on the English held-out sentences, where chance would give about 19. What does that say?",
        options={
            "combine": "Combining the two would fix most of their mistakes",
            "right": "Much of what goes wrong lies in the sentences, not in either method",
            "copied": "The lexicon learned its mistakes from the classifier",
            "chance": "Little: with more than 100 mistakes each, that much overlap happens by chance",
        },
        correct="right",
        why={
            "right": "Two methods that work in very different ways go wrong on the same sentences: a label that the sentence does not explain, advice such as *Don't miss it.*, a word that means something else here than in most reviews. Both read single words, and these sentences say something that their words, one by one, do not. A better method of the same kind will not fix them; a classifier that reads words in their context, such as the BERT of *BERT: filling in the blank*, has a better chance, and some labels no classifier will get right.",
            "combine": "Where both are wrong, they give the same wrong label: each gives 1 or 0, so in all 74 sentences the two agree. A vote of the two says the same there. Combining can only help in the 67 sentences where exactly one of the two is right.",
            "copied": "The lexicon comes from the counts of each word in the training sentences; it never sees the classifier. The overlap comes from the sentences: both learn from the same training sentences, and both are tested on the same held-out ones.",
            "chance": "By chance, about $109 \\times 106 / 600 \\approx 19$ sentences would be wrong in both. 74 is almost four times as many, so chance alone does not explain the overlap.",
        },
    )
    choice_shared
    return (choice_shared,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Project: a spam filter with TF-IDF](/course/open/nlp-project-spam?lang=en) gave your project a classifier and the code to train it; here the same code learned sentiment, in English and in Dutch, without a line changed. [A list of good and bad words](/course/open/nlp-sentiment-lexicon?lang=en) gave the lexicon, now with a label for each sentence and no label where it says nothing. [Many classes: micro and macro averages](/course/open/nlp-micro-macro?lang=en) gave macro F1, now with a missing label counted as a false negative. On these reviews, the two classifiers end up about equally good, and they share most of their mistakes: what remains lies in sentences that single words do not explain. With the spam filter, your project `text` now holds `logistic.py`, `features.py`, `train.py`, `judge.py`, `lexicon.py`, `scores.py` and `compare.py`. The next lesson, *What is this collection about? Topic models*, finds what a collection of documents is about without any labels at all.

    **An everyday example.** Two friends give a film a thumbs up or down, one from the trailer, the other from the reviews. When both are wrong about the same film, the film was probably hard to judge from the outside: a slow start, a joke you only get at the end. Asking a third friend who also only saw the trailer will not help.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shared, result_compare, result_lexicon_labels, result_macro_f1):
    L.wrap_up(result_lexicon_labels, result_macro_f1, result_compare, choice_shared)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
