# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-sts. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-sts", "en")
    C = L.checks   # the pairs of the STS Benchmark with Granite's cosine similarities, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions, about [A vector for a whole sentence](/course/open/nlp-sentence-embeddings?lang=en) and [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sentence():
    quick_sentence = L.choice(
        "quick-sentence",
        question="In *A vector for a whole sentence*, the mean of the GloVe vectors gave *the dog bit the man* and *the man bit the dog* exactly the same vector. Why?",
        options={
            "same": "The two sentences mean the same",
            "length": "Each mean has a length of 1",
            "right": "A mean does not see the order of the words",
            "glove": "GloVe gives all the words of a sentence one shared vector",
        },
        correct="right",
        why={
            "right": "Both sentences have the same five words, and a mean adds them up in any order. So the same vector does not mean the same meaning: this lesson compares cosine similarities with what people say about the meaning.",
            "same": "They do not: in one the dog bites, in the other the man. Their means are the same because both have the same five words, each as many times, and a mean does not see their order.",
            "length": "A mean of word vectors need not have length 1, and a length says nothing about which words went in. Both sentences have the same five words, and a mean does not see their order.",
            "glove": "GloVe gives every word its own vector, the same in every sentence. The mean of those vectors is the same for both sentences, because both have the same five words, and a mean does not see their order. See [A vector for a whole sentence](/course/open/nlp-sentence-embeddings?lang=en).",
        },
    )
    quick_sentence
    return (quick_sentence,)


@app.cell(hide_code=True)
def quick_check_precision():
    quick_precision = L.choice(
        "quick-precision",
        question="In *Beyond accuracy: precision and recall*, a spam filter calls 10 mails spam, and 8 of them are spam. There are 20 spam mails in all. What are its precision and recall?",
        options={
            "both": "Both 0.8",
            "right": "Precision 0.8, recall 0.4",
            "swap": "Precision 0.4, recall 0.8",
            "unknown": "Precision 0.8, and recall needs the other mails as well",
        },
        correct="right",
        why={
            "right": "Of the 10 mails it calls spam, 8 are: precision 8 / 10. Of the 20 spam mails, it finds 8: recall 8 / 20. This lesson computes both for a threshold on the cosine similarity.",
            "both": "Precision is 8 / 10, of the mails the filter calls spam. Recall is a share of the 20 mails that are spam: how many of those does it find?",
            "swap": "The two are the other way around. Precision is a share of the 10 mails the filter calls spam, and recall a share of the 20 spam mails. See [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
            "unknown": "Recall is a share of the spam mails only: the filter finds 8 of the 20, so recall is 0.4. The other mails count only for the false positives, and those are in the precision.",
        },
    )
    quick_precision
    return (quick_precision,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A forum that spots a question that was asked before, a tool that checks short answers to an exam question, a check of whether a translation kept the meaning: each needs to know whether two sentences say the same thing, in whatever words. Daniel Cer and others, who ran the STS task of 2017, list such uses: "machine translation (MT), summarization, generation, question answering (QA), short answer grading, semantic search, dialog and conversational systems".

    A cosine similarity of two sentence embeddings is a number, not yet an answer. This lesson compares those numbers with what people judged, and turns them into a yes or a no: the same meaning, or not.

    **An everyday example.** At a lost-and-found desk, *a black umbrella with a wooden handle* and *a dark umbrella, handle made of wood* are the same umbrella. *A black umbrella with a red handle* shares more words with the first, and is another umbrella.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_one = L.guess(
        "guess",
        question="The STS Benchmark holds pairs of sentences that people scored for how alike they are in meaning, from 0, completely dissimilar, to 5, the same meaning. This lesson uses 1,750 of those pairs: short sentences about photos. People gave 167 of them a 0. What cosine similarity does Granite Embedding give those 167 pairs on average?",
        options={
            "zero": "About 0, as their meanings share nothing",
            "low": "About 0.2",
            "right": "About 0.56",
            "high": "About 0.9",
        },
        reveal="**About 0.56:** 0.560 on average, and not one of the 167 lower than 0.325. *A yellow school bus parked in a field.* and *A brown horse in a green field.* got a 0 from people and 0.772 from Granite: the two share *in a field*, and both describe a photo. A cosine similarity of 0.56 does not mean *half the same*. What counts is how a pair compares with other pairs, and the idea shows how.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### How alike in meaning: STS

    **Semantic textual similarity**, STS for short, is how alike two sentences are in meaning, as a number. In the words of Cer and others: "Semantic Textual Similarity (STS) assesses the degree to which two sentences are semantically equivalent to each other." From 2012 to 2017, the STS tasks of SemEval, a series of contests in which teams compare their systems on the same data, asked people to score pairs of sentences on a scale from 0 to 5. Cer and others give the scale with these explanations, from Agirre and others (2013):

    | Score | Explanation |
    |---|---|
    | 5 | "The two sentences are completely equivalent, as they mean the same thing." |
    | 4 | "The two sentences are mostly equivalent, but some unimportant details differ." |
    | 3 | "The two sentences are roughly equivalent, but some important information differs/missing." |
    | 2 | "The two sentences are not equivalent, but share some details." |
    | 1 | "The two sentences are not equivalent, but are on the same topic." |
    | 0 | "The two sentences are completely dissimilar." |

    For the pairs of 2017, paid workers on Amazon Mechanical Turk scored each pair. Cer and others write: "Five annotations are collected per pair." And: "Gold scores average the five individual annotations." So a score can lie between two whole numbers, such as 3.4.

    The **STS Benchmark** collects 8,628 English pairs of those tasks, from 2012 to 2017, with their scores: from news, from forums, and captions of photos and videos. Only two of its sources let the course pass their sentences on with a short credit: captions of photos of the PASCAL collection, scored in 2014 and 2015, and sentences about photos from SNLI, scored in 2017. Those are the 1,750 pairs of this lesson. Beside each English sentence the course has a Dutch one: a machine translation by DeepL, made by Philip May in 2021. The scores belong to the English pairs, so this lesson computes with the cosine similarities of the English sentences; the Dutch translations come back under *Watch out*, in the last exercise and when you try it yourself.
    """)
    return


@app.cell(hide_code=True)
def idea_scatter():
    mo.Html(C.idea_scatter_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_scatter():
    mo.md(r"""
    The higher the score, the higher Granite's cosine similarity on average: from 0.580 for the pairs scored under 0.5 to 0.926 for the pairs of 4.5 and more. But the clouds overlap. *The gate is blue.* and *The gate is yellow.* get 0.951 with a score of 1.6, more than *A man is laughing with a woman* and *A man and a woman laughing.* get with a score of 4.8: 0.786. Two sentences that differ in one word get a high cosine similarity from Granite, also when that one word changes what the photo shows.
    """)
    return


@app.cell(hide_code=True)
def idea_examples():
    mo.Html(C.idea_examples_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_paraphrase():
    mo.md(r"""
    ### A paraphrase, and where it begins

    A **paraphrase** of a sentence says the same thing in other words. The STS scale does not say where a paraphrase begins; this lesson draws the line at 4: a pair with a score of 4 or more counts as a paraphrase, its two sentences equivalent apart from unimportant details. **Paraphrase detection**, deciding of a pair whether it is a paraphrase, is then a classification with two classes.

    With sentence embeddings, such a classifier needs one number: a threshold on the cosine similarity. A pair counts as a paraphrase when its cosine similarity is at least the threshold, as a mail counted as spam from a probability of 0.5 on in [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en). Each pair then lands in one cell of the confusion matrix: a paraphrase that counts as one is a true positive, a paraphrase below the threshold a false negative, another pair that counts as a paraphrase a false positive. Precision and recall follow as in that lesson.

    The threshold is chosen on one set of pairs and measured on another, as in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en): a threshold that just reaches a precision on one set is tuned to the pairs of that set, and only pairs it was not chosen on show what it gives on new pairs. Granite learned elsewhere, so nothing here learns from the STS Benchmark's train part: with its dev part, it makes this lesson's validation set, 1,375 pairs, of which 352 are paraphrases. Its test part, 375 pairs with 91 paraphrases, is the test set: it measures the chosen threshold once, at the end.
    """)
    return


@app.cell(hide_code=True)
def idea_threshold():
    mo.Html(C.idea_threshold_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_choose():
    mo.md(r"""
    At the threshold 0.85, 494 pairs of the validation set count as a paraphrase, and 291 of them are one: precision 0.589. Of the 352 paraphrases, the threshold finds those 291: recall 0.827. A higher threshold never counts more pairs, so recall never goes up, and precision usually does, as with the spam filter.

    Which threshold is right depends on which mistake costs more. A forum that merges two questions when they ask the same thing should rarely merge two different questions: it needs a high precision. And of all thresholds that reach that precision, the lowest keeps the most recall, because recall never goes up as the threshold goes up.
    """)
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **semantic textual similarity (STS)** | how alike two sentences are in meaning, as a number; in the STS tasks, a score of people from 0 to 5 |
        | **STS Benchmark** | 8,628 English pairs of sentences of the STS tasks of 2012 to 2017, each with its score |
        | **paraphrase** | a sentence that says the same thing as another in other words; in this lesson, a pair with a score of 4 or more |
        | **paraphrase detection** | deciding of a pair of sentences whether it is a paraphrase |
        | `cos[score >= 4]` | the numbers of `cos` at the places where `score` is 4 or more, with a boolean mask as in [Selecting parts of an array](/course/open/np-indexing?lang=en) |
        | `sorted(cos)` | the numbers of the array `cos` as a new list, from low to high |
        """),
        L.watch_out("The Dutch sentences are machine translations, and their scores are those of the English pairs. A translation can change a pair: DeepL made *gate* into *hek* in one sentence and into *poort* in the other, so *Het hek is blauw.* and *De poort is geel.* differ in more than a color. Philip May checked nine Dutch pairs by hand, and noted one translation that is partly English and one with poor grammar. A number measured on the Dutch pairs measures Granite and the translation together."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Take pairs of sentences with scores of people, in a validation set and a test set.
    2. Compute the cosine similarity of the sentence embeddings of each pair.
    3. Draw the line for a paraphrase on the score scale; here at 4.
    4. On the validation set, compute precision and recall for each threshold, and choose the threshold that fits what a mistake costs: for a precision you need, the lowest threshold that reaches it.
    5. Measure precision and recall of that threshold once, on the test set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.sts("validation", "en")` gives two arrays of 1,375 numbers: `cos`, the cosine similarity of Granite's embeddings of the two English sentences of each pair of the validation set, and `score`, the score people gave the pair. `C.sts("test", "en")` gives the 375 pairs of the test set, and `"nl"` instead of `"en"` the cosine similarities of the Dutch translations, with the same scores. The code counts the paraphrases, takes the mean cosine similarity of the paraphrases and of the other pairs, and counts the pairs that reach the threshold 0.85.
    """)
    return


@app.cell
def example_cell():
    import numpy as np

    cos, score = C.sts("validation", "en")
    print(len(cos), np.sum(score >= 4))
    print(np.mean(cos[score >= 4]))
    print(np.mean(cos[score < 4]))
    print(np.sum(cos >= 0.85))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Of the 1,375 pairs, 352 are paraphrases. They get 0.899 on average, the other pairs 0.731: far apart on average, but the two clouds overlap, as in the picture. 494 pairs reach the threshold 0.85, among them 203 that are not paraphrases.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Where on the scale?

    Each line holds a pair of sentences. Put each pair under the scores that people would give it on the STS scale of the idea: 4 or 5, 1 to 3, or 0.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_scale = L.sort("sts-scale")
    sort_scale
    return (sort_scale,)


@app.cell(hide_code=True)
def check_sort(sort_scale):
    result_sort = L.check("sts-scale", sort_scale)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_after_sort():
    mo.md(r"""
    The STS scale asks what two sentences say. A cosine similarity can stay high when one word changes what a sentence says, as with the gate that is blue or yellow in the idea: a threshold on that number has to live with that. The next two exercises measure how well it does.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pr():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. precision_recall: what a threshold gets right

        Write `precision_recall(cos, score, threshold)`: `cos` holds the cosine similarity of each pair, and `score` its score. A pair is a paraphrase when its score is 4 or more, and counts as one when its cosine similarity is at least `threshold`. Give back two values: the precision first, then the recall. A share that would divide by 0 is `None`: the precision when no pair counts as a paraphrase, the recall when no pair is one.

        For `precision_recall(np.array([0.9, 0.8, 0.7, 0.6]), np.array([4.5, 2.0, 4.0, 5.0]), 0.75)` the precision is 0.5 and the recall 1 / 3: two pairs reach 0.75, and one of them is a paraphrase; of the three paraphrases, the threshold finds one. With the threshold 0.95, no pair counts, so the precision is `None` and the recall 0. The check calls both, then cases of its own: a pair exactly at the threshold, pairs without a single paraphrase, and `cos` and `score` of `C.sts("validation", "en")` and of the test set.
        """),
        L.task("Give back the precision and the recall of the threshold, with None for a share that would divide by 0."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pr():
    box_pr = L.code_box("precision_recall", lines=16)
    box_pr
    return (box_pr,)


@app.cell(hide_code=True)
def check_pr(box_pr):
    result_pr = L.check("precision_recall", box_pr.value)
    result_pr
    return (result_pr,)


@app.cell(hide_code=True)
def hints_pr():
    L.hints(
        "precision_recall",
        [
            "Split the cosine similarities with a boolean mask: `para`, those of the pairs with a score of 4 or more, and `other`, those of the rest. Count `tp`, `fp` and `fn` with `np.sum` and a comparison with `threshold`, as with the spam filter. Then start `precision` and `recall` at `None`, and compute each only when what it divides by is more than 0.",
            "A true positive is a paraphrase at or above the threshold: `np.sum(para >= threshold)`. Which array and which comparison give the false positives, and which the false negatives?",
            r"`    if tp + fp > 0:` with `precision = tp / (tp + fp)` under it, and for the recall `    if ___ > 0:`",
        ],
        why="`para = cos[score >= 4]` and `other = cos[score < 4]`; then `tp = np.sum(para >= threshold)`, `fp = np.sum(other >= threshold)` and `fn = np.sum(para < threshold)`. Start `precision = None` and `recall = None`; `if tp + fp > 0:` sets `precision = tp / (tp + fp)`, and `if tp + fn > 0:` sets `recall = tp / (tp + fn)`. Last, `return precision, recall`. For the example: `para` is `[0.9, 0.7, 0.6]` and `other` is `[0.8]`, so `tp` is 1, `fp` 1 and `fn` 2: precision 1 / 2, recall 1 / 3.",
    )
    return


@app.cell(hide_code=True)
def text_after_pr():
    mo.md(r"""
    In the chart, your recall goes down as the threshold goes up, and never rises again. Your precision goes up, with small dips, from 0.349 at 0.70 to 1 near the top. At the highest cosine similarity of the validation set, 0.9942, only one pair counts: a paraphrase, so precision 1 and recall 1 / 352. Above every pair, at 1.00, nothing counts, and precision is `None`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_low():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. lowest_threshold: the lowest threshold that is precise enough

        Write `lowest_threshold(cos, score, target)`: the lowest threshold whose precision is at least `target`, or `None` when no threshold reaches it. Try the cosine similarities of the pairs themselves as thresholds, from low to high: any other threshold counts the same pairs as the next cosine similarity above it. The box may use `precision_recall`: the course's version, which works as yours should.

        For `lowest_threshold(np.array([0.8, 0.6, 0.9, 0.7]), np.array([4.0, 1.0, 5.0, 4.2]), 0.8)` the result is `0.7`: at 0.6 all four pairs count and three are paraphrases, precision 0.75; at 0.7 three count and all three are paraphrases, precision 1. With the target 0.75 the result is `0.6`. The check calls both, then cases of its own: pairs where no threshold reaches the target, and the validation sets of the English pairs and of the Dutch translations.
        """),
        L.task("Give back the lowest cosine similarity whose precision reaches target, or None."),
    ])
    return


@app.cell(hide_code=True)
def exercise_low():
    box_low = L.code_box("lowest_threshold", lines=10)
    box_low
    return (box_low,)


@app.cell(hide_code=True)
def check_low(box_low):
    result_low = L.check("lowest_threshold", box_low.value)
    result_low
    return (result_low,)


@app.cell(hide_code=True)
def hints_low():
    L.hints(
        "lowest_threshold",
        [
            "Step 4 of the recipe: try the cosine similarities of the pairs as thresholds `t`, from low to high, with `sorted(cos)`, and ask `precision_recall` for the precision of each. The first `t` with enough precision is then the lowest; when none has enough, give back `None`.",
            "Write `for t in sorted(cos):` with `precision, recall = precision_recall(cos, score, t)` in it. When should the loop give back `t`? Read the task again: what must the precision be?",
            r"`        if ___:` with `return t` under it",
        ],
        why="`for t in sorted(cos):` with `precision, recall = precision_recall(cos, score, t)` in it, then `if precision >= target:` with `return t` under it; after the loop, `return None`. For the example: at 0.6 the precision is 0.75, less than 0.8; at 0.7 it is 1, so the function gives back 0.7.",
    )
    return


@app.cell(hide_code=True)
def text_after_low():
    mo.md(r"""
    For a precision of 0.8 on the English pairs, your function chooses 0.9418, and that costs recall: 0.233, under a quarter of the paraphrases. On the test set, the same threshold gives a precision of 0.875 and a recall of 0.385: other pairs, other numbers, though the threshold carries over. On the test set only 40 pairs reach the threshold, so one pair more or less moves the precision by about 0.02. For 0.9, the threshold goes up to 0.9555, and recall down to 0.156. Many of the pairs that count without being a paraphrase lie just under the line: at 0.9418, 20 such pairs count, and 11 of them got 3.6 or more from people. On the Dutch translations, a precision of 0.8 takes the threshold 0.9318, with a recall of 0.270.
    """)
    return


@app.cell(hide_code=True)
def text_explore_threshold():
    mo.md(r"""
    ### Try it: move the threshold

    Choose a threshold, and the English pairs or the Dutch translations of the validation set. The picture shows the pairs, the threshold, the confusion matrix, precision and recall. Look for the threshold where recall starts to fall fast, and compare the two languages at the same threshold.
    """)
    return


@app.cell(hide_code=True)
def explore_threshold_controls():
    explore_t = mo.ui.slider(0.70, 1.00, step=0.005, value=0.85, label="Threshold", show_value=True)
    explore_data = mo.ui.radio(options={"English pairs": "en", "Dutch translations": "nl"}, value="English pairs", label="Pairs")
    mo.hstack([explore_t, explore_data], justify="start", gap=2)
    return explore_data, explore_t


@app.cell(hide_code=True)
def explore_threshold(explore_data, explore_t):
    mo.Html(C.explore_picture(explore_t.value, explore_data.value, L.lang))
    return


@app.cell(hide_code=True)
def text_explore_granite():
    mo.md(r"""
    ### Try it: your sentences in Granite

    Granite Embedding runs on your computer once it is downloaded. Type up to six sentences, one per line: the picture compares every two, the most alike first, and says which pairs reach the threshold that your target precision gives on the English pairs of the validation set. Try a sentence and its paraphrase, the same sentence with *not*, and a sentence about something else. The threshold comes from captions of photos: does it fit your sentences?
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download Granite Embedding (223 MB)")
    return (get_button,)


@app.cell(hide_code=True)
def granite_download(get_button):
    C.granite_or_stop(get_button, {
        "ask": "*This part needs Granite Embedding, which is not on your computer yet. The button downloads it once from Hugging Face, at a fixed version: `ibm-granite/granite-embedding-107m-multilingual`, 223 MB, under the Apache-2.0 license. The exercises and the rest of the lesson work without it.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: a file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is not enough room on your disk. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    granite_here = True
    mo.md("*Granite Embedding is downloaded.*")
    return (granite_here,)


@app.cell(hide_code=True)
def explore_granite_controls():
    explore_lines = mo.ui.text_area(value="The museum opens at ten.\nAt ten, the museum opens its doors.\nThe museum does not open at ten.\nThe soup is too salty.", label="Your sentences, one per line", rows=6, full_width=True)
    explore_target = mo.ui.dropdown(options=C.TARGET_CHOICES, value="0.8", label="Target precision")
    mo.vstack([explore_lines, explore_target])
    return explore_lines, explore_target


@app.cell(hide_code=True)
def explore_granite(explore_lines, explore_target, granite_here):
    _ = granite_here
    mo.Html(C.granite_picture(explore_lines.value, explore_target.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [A vector for a whole sentence](/course/open/nlp-sentence-embeddings?lang=en) gave every sentence one vector, and two sentences a cosine similarity. Here people's scores judge those numbers: on average they rise with the score, but a pair that differs in one important word can come out higher than a paraphrase. A threshold turns the number into paraphrase detection, with precision and recall as in [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en), chosen on the validation set and measured on the test set as in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en). The next lesson of the domain, *Choosing a sentence model: reading MTEB*, compares sentence models on many tasks, of which STS is one.

    **An everyday example.** A metal detector on a beach beeps at every coin, and at every bottle cap too. Turn it down and it beeps only for large pieces of metal: fewer caps, and more coins left in the sand. Where to set it depends on what a cap costs, and what a coin is worth.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_low, result_pr, result_sort):
    L.wrap_up(result_sort, result_pr, result_low)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
