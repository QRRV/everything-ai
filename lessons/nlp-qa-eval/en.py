# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-qa-eval. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-qa-eval", "en")
    C = L.checks   # the questions of SQuAD and SQuAD-NL with every answer people marked, the scores, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Finding the answer in a passage](/course/open/nlp-extractive-qa?lang=en), and one about [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_people():
    quick_people = L.choice(
        "quick-people",
        question="In *Finding the answer in a passage*, each answer of SQuAD is a span that people marked in a paragraph of Wikipedia. For a question of the development set, how many people marked an answer?",
        options={
            "one": "One: the person who wrote the question",
            "right": "At least three: the writer of the question and at least two more",
            "program": "None: a program marked the answers, and people only checked them afterwards",
            "two": "Exactly two, so that one can check the other",
        },
        correct="right",
        why={
            "right": "For the development set and the test set, at least two more people were each asked to mark an answer to every question, besides the person who wrote it. Their spans often differ a little, and this lesson scores an answer against all of them.",
            "one": "That holds for the training set. What did SQuAD add for the questions of the development set? See [Finding the answer in a passage](/course/open/nlp-extractive-qa?lang=en).",
            "program": "People on a crowdsourcing platform highlighted the answers themselves. How many of them for each question of the development set? See [Finding the answer in a passage](/course/open/nlp-extractive-qa?lang=en).",
            "two": "There were more than two. Who marked an answer besides the person who wrote the question? See [Finding the answer in a passage](/course/open/nlp-extractive-qa?lang=en).",
        },
    )
    quick_people
    return (quick_people,)


@app.cell(hide_code=True)
def quick_check_f1():
    quick_f1 = L.choice(
        "quick-f1",
        question="In *Precision and recall in one number: the F1 score*, the F1 score is the harmonic mean of precision and recall. A detector has a precision of 0.5 and a recall of 1. What is its F1 score?",
        options={
            "mean": "0.75",
            "min": "0.5",
            "right": "About 0.67",
            "one": "1, as its recall is perfect",
        },
        correct="right",
        why={
            "right": "Twice the product, divided by the two added up: 2 × 0.5 × 1 / (0.5 + 1) = 1 / 1.5, about 0.67. The harmonic mean stays close to the smaller of the two. This lesson works out precision and recall over the words of an answer.",
            "mean": "0.75 is the plain mean of 0.5 and 1. Which mean stays close to the smaller number? See [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en).",
            "min": "0.5 is the smaller of the two. Is the harmonic mean of 0.5 and 1 exactly the smaller one? See [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en).",
            "one": "The F1 score is 1 only when both are 1. What does a precision of 0.5 do to it? See [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en).",
        },
    )
    quick_f1
    return (quick_f1,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A teacher who marks a test compares each answer with the answer key. The key says *1850*; one pupil writes *in 1850*, another *the year 1850*, and the teacher gives both the point. A computer that scores thousands of answers cannot read along, so it needs a rule for when an answer counts as right, and for how close a nearly right answer comes. SQuAD came with two such rules, exact match and F1, and systems for reading comprehension have been scored with them ever since.

    In this lesson you write both, and measure with them how well people agree with each other, and how far a whole sentence, the answer of your reader from *Finding the answer in a passage*, is from the answer.

    **An everyday example.** At a pub quiz, the answer sheet says *Amsterdam*. A team writes *the city of Amsterdam*: most quiz masters give the point, a strict one does not, and some give half a point.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_people = L.guess(
        "guess",
        question="In SQuAD, several people marked an answer to each question of the development set. Take the answer of the second person, and compare it with the answers of the others, leaving out capitals, punctuation and the words *a*, *an* and *the*. For how many of the 1,190 English questions is it exactly the same as at least one of them?",
        options={
            "half": "About half",
            "two-thirds": "About two thirds",
            "right": "About four in five",
            "all": "Almost every one of them",
        },
        reveal="About four in five: 980 of the 1,190, or 82.4%. People who read the same paragraph and the same question still mark a different span for about one question in six: one marks *between 1500 and 1850*, another *1500 and 1850*. They mostly agree on the answer, but not on how much to mark around it. With partial credit for the words two answers share, the F1 score of this lesson, people score 91.8%.",
    )
    guess_people
    return (guess_people,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    The answer of a reader is a span, a piece of the paragraph. To score it, compare it with the **reference answers**: the spans that people marked for the same question. In the development set of SQuAD, the person who wrote a question marked an answer, and at least two more people were each asked to mark one, and could leave a question without an answer when the paragraph did not answer it; 1,175 of the 1,190 questions of this lesson have three reference answers or more, the other 15 two. Rajpurkar and others scored an answer in two ways, and wrote: "Both metrics ignore punctuations and articles (a, an, the)."

    **Exact match** counts an answer as right when it is the same as one of the reference answers, once both are normalized: in small letters, without punctuation and without the articles. In their words: "This metric measures the percentage of predictions that match any one of the ground truth answers exactly." So *The Horniman* matches *Horniman*, but *in 2018* does not match *2018*, though both say the same.

    **Token F1** gives partial credit for the words an answer and a reference answer share. "We treat the prediction and ground truth as bags of tokens, and compute their F1." A bag is like a set in which a word can be more than once; a word that both have twice is shared twice. Precision is the share of the words of the answer that are shared, recall the share of the words of the reference answer, and F1 their harmonic mean, as in *Precision and recall in one number*. An answer with too many words loses precision, an answer with words missing loses recall.

    A question has several reference answers, and SQuAD takes the best: "We take the maximum F1 over all of the ground truth answers for a given question, and then average over all of the questions." In the same way, exact match counts a question as right when its answer matches any one of them. The picture scores one answer to a question about the plague against its three reference answers.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **reference answer** | an answer that people marked, with which the answer of a system is compared; SQuAD calls them *ground truth answers* |
    | **exact match** (EM) | an answer counts as right only when it is the same as a reference answer, once both are normalized; over many questions, the share that is right |
    | **token F1** | the F1 score of the words of an answer against the words of a reference answer: precision over the answer, recall over the reference answer |
    | **bag** | a collection like a set, in which an item can be more than once: *pool pool* holds *pool* twice |
    | **articles** | *a*, *an* and *the* in English; *de*, *het* and *een* in Dutch |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To score the answers of a reader the way SQuAD does:

    1. Normalize the answer and each reference answer: the words in small letters, without punctuation and without *a*, *an* and *the*.
    2. Exact match: 1 when the answer has the same words, in the same order, as at least one reference answer; else 0.
    3. Token F1 against one reference answer: count the shared words, each as often as it is in both. With none shared, F1 is 0. Else precision = shared / words of the answer, recall = shared / words of the reference answer, and F1 = 2 × precision × recall / (precision + recall).
    4. For each question, take the best exact match and the best F1 over its reference answers.
    5. Take the mean of each over the questions, as a percentage.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Take the question *During which years was the plague present in Islamic countries?* Its reference answers are *between 1500 and 1850*, *1500 and 1850*, and *1500 and 1850* again. Suppose a reader marks *virtually every year between 1500 and 1850*: right, but with three words too many. It is not the same as any reference answer, so its exact match is 0. With the first reference answer it shares four words, *between*, *1500*, *and* and *1850*. A few lines work out precision, recall and F1; `C.words` is the `words` of *Kinds of question answering*:
    """)
    return


@app.cell
def example_cell():
    _answer = C.words("virtually every year between 1500 and 1850")
    _reference = C.words("between 1500 and 1850")
    print(_answer)
    print(_reference)
    _precision = 4 / len(_answer)
    _recall = 4 / len(_reference)
    print(_precision, _recall, 2 * _precision * _recall / (_precision + _recall))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The answer has 7 words and 4 of them are shared, so the precision is 4/7 = 0.57. All 4 words of the reference answer are in the answer, so the recall is 1. The F1 score is 2 × 0.57 × 1 / 1.57 = 0.73. Against *1500 and 1850*, the precision is 3/7 and the F1 score 0.6. SQuAD takes the best: for this question, exact match 0 and F1 0.73. The three words too many cost a quarter of the score; the whole sentence, with more words still, would cost more.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_match():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. exact_match: the same answer?

        Write `exact_match`, which gets `prediction`, the answer of a reader, `answer`, a reference answer, and `articles`, a set of articles such as `C.ARTICLES["en"]`, which holds *a*, *an* and *the*. Give back 1 when the two have the same words in the same order once both are normalized, and else 0. The box knows `words`, which gives the words of a text in lower case, without punctuation. In the start code, `p` should hold the words of `prediction` without the articles, and `a` those of `answer`; the lines after them compare the two. Replace the two `...`.

        The check calls `exact_match("The pool.", "pool", C.ARTICLES["en"])`, which gives `1`. The same in Dutch, with `C.ARTICLES["nl"]`: *de*, *het* and *een*. Then cases of its own: words in another order, a word too many, a capital or a full stop, a word that comes twice, *the* inside a longer word, and answers of SQuAD.
        """),
        L.task("Say whether an answer is the same as a reference answer, once both are normalized."),
    ])
    return


@app.cell(hide_code=True)
def exercise_match():
    box_match = L.code_box("exact_match", lines=6)
    box_match
    return (box_match,)


@app.cell(hide_code=True)
def check_match(box_match):
    result_match = L.check("exact_match", box_match.value)
    result_match
    return (result_match,)


@app.cell(hide_code=True)
def hints_match():
    L.hints(
        "exact_match",
        [
            "`p` is a list comprehension over the words of `prediction`, as `words` gives them, that keeps each word `w` that is not in `articles`; `a` is the same for `answer`. Two lists are equal with `==` when they hold the same items in the same order.",
            "Keep the words as a list, not as a set: a set forgets the order, so *1850 and 1500* would match *1500 and 1850*. And use `words`, not `split`: `split` keeps the capitals and the full stops.",
            "`    a = [w for w in ___ if w not in articles]`",
        ],
        why="`p = [w for w in words(prediction) if w not in articles]` and `a = [w for w in words(answer) if w not in articles]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_match():
    mo.md(r"""
    Normalizing matters. Of the 1,190 English questions, the answer of the second person is the very same text as an answer of another person for 76.3% of them. Without capitals and punctuation 78.7% match, and without the articles as well 82.4%. The script of SQuAD normalizes the same way, with one difference: it deletes the punctuation inside a word and joins the two parts, so *27-30* becomes *2730*, where `words` cuts it into *27* and *30*. On these questions, that changes no exact match, and moves the F1 score of people from 91.8% to 91.6%.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_f1():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. token_f1: partial credit

        Write `token_f1`, which gets `prediction`, `answer` and `articles` as in exercise 1, and gives back the F1 score of the words of `prediction` against the words of `answer`. The start code makes `p` and `a` as in exercise 1, and `same`, the number of words they share, with the course's `shared`, which counts a word as often as it is in both: `shared(["pool", "pool"], ["pool"])` gives `1`. When they share no word, precision and recall are both 0 and the F1 score would divide by 0, so the start code gives back `0.0` first. Replace the three `...`: `precision`, `recall` and the F1 score.

        The check calls `token_f1("at 9:00", "9:00", C.ARTICLES["en"])`, which gives `0.8`: *9* and *00* are shared, so the precision is 2/3 and the recall 1. The same in Dutch, and then cases of its own: an answer with too many words, an answer with words missing, a word that comes twice, and the answers of the worked example.
        """),
        L.task("Work out the F1 score of the words an answer shares with a reference answer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_f1():
    box_f1 = L.code_box("token_f1", lines=10)
    box_f1
    return (box_f1,)


@app.cell(hide_code=True)
def check_f1(box_f1):
    result_f1 = L.check("token_f1", box_f1.value)
    result_f1
    return (result_f1,)


@app.cell(hide_code=True)
def hints_f1():
    L.hints(
        "token_f1",
        [
            "`precision` is `same` divided by the number of words in `p`, with `len`; `recall` is `same` divided by the number of words of the reference answer. The `return` gives their harmonic mean, as in *Precision and recall in one number*: twice the product of `precision` and `recall`, divided by `precision` plus `recall`.",
            "Count words, not characters: `len(prediction)` is the number of characters of the text, `len(p)` the number of words. And put brackets around `precision + recall`: Python divides before it adds.",
            "`    recall = same / ___`",
        ],
        why="`precision = same / len(p)`, `recall = same / len(a)` and `return 2 * precision * recall / (precision + recall)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_f1():
    mo.md(r"""
    When the answers of two people differ, they usually still share words. Of the 1,190 English questions, the answer of the second person matches none of the others exactly for 210. For 171 of those it shares words with at least one of them, and only for 39 does it share none. Rajpurkar and others found the same: "Mismatch occurs mostly due to inclusion/exclusion of non-essential phrases (e.g., monsoon trough versus movement of the monsoon trough) rather than fundamental disagreements about the answer." The people who added answers were "asked to select the shortest span in the paragraph that answered the question". On the question about the plague, the first answer is *between 1500 and 1850*, and the two answers after it are *1500 and 1850*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_scores():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. squad_scores: the scores over many questions

        Write `squad_scores`, which gets `predictions`, the answers of a reader to a list of questions, `answer_lists`, for each question the list of its reference answers, and `articles`. Give back the exact match and the F1 score over all the questions, in percent: for each question the best score of its answer over its reference answers, and then the mean over the questions. The box knows the course's `exact_match` and `token_f1`, so your answers to exercises 1 and 2 do not matter here. In the start code, `em` and `f1` add up the best score of each question, and the `return` turns the totals into percentages. Replace the two `...`.

        The check calls `squad_scores(["at 9:00", "the pool"], [["9:00"], ["the big pool", "pool"]], C.ARTICLES["en"])`, which gives `(50.0, 90.0)`: *at 9:00* matches no reference answer and has F1 0.8, and *the pool* matches *pool* exactly. The same in Dutch, and then all questions with the sentences that the reader of *Finding the answer in a passage* picks as answers (in the lesson: `C.reader_answers("en")` and `C.answer_lists("en")`), and the answers of people.
        """),
        L.task("Score many answers the way SQuAD does."),
    ])
    return


@app.cell(hide_code=True)
def exercise_scores():
    box_scores = L.code_box("squad_scores", lines=8)
    box_scores
    return (box_scores,)


@app.cell(hide_code=True)
def check_scores(box_scores):
    result_scores = L.check("squad_scores", box_scores.value)
    result_scores
    return (result_scores,)


@app.cell(hide_code=True)
def hints_scores():
    L.hints(
        "squad_scores",
        [
            "For question `i`, the answer is `predictions[i]`, and its reference answers are at place `i` of `answer_lists`. A list comprehension gives `exact_match` of the answer with each reference answer `a`, and `articles`; `max` takes the best of them, and `em` goes up by it. The line of `f1` does the same with `token_f1`.",
            "Take the best, not the first reference answer: *the pool* matches *pool*, the second reference answer, exactly. And keep one score per question: a loop over the reference answers that adds every score counts a question more than once.",
            "`        em = em + max([exact_match(predictions[i], a, articles) for a in ___])`",
        ],
        why="`em = em + max([exact_match(predictions[i], a, articles) for a in answer_lists[i]])` and `f1 = f1 + max([token_f1(predictions[i], a, articles) for a in answer_lists[i]])`.",
    )
    return


@app.cell(hide_code=True)
def text_after_scores():
    mo.md(r"""
    People score 82.4% exact match and 91.8% F1 against each other on these questions; on the whole development set, Rajpurkar and others measured 80.3% and 90.5% this way. The sentence that holds the answer scores 0.5% and 24.5%. Its recall is almost always 1, as it holds the whole answer, but it has 28.0 words without the articles, against about 3 for an answer, so its precision is low. The sentence that your reader picks scores 19.5% F1, close to the 20.2% of the sliding window, a simple baseline of the paper, and the first sentence of the paragraph 9.8%. The logistic regression of the paper, which marks a span, reached 51.0% F1: most of the score lies in marking the span.

    ### Try it: score an answer

    Pick a question, type an answer, and see how it scores against each reference answer that people marked. As long as you type nothing, the box scores the sentence that the reader of *Finding the answer in a passage* picks. The box leaves out the paragraphs that the course does not show: it has 876 of the 1,190 questions.
    """)
    return


@app.cell(hide_code=True)
def box_number():
    try_n = mo.ui.slider(1, C.box_count(L.lang), value=1, label="Question", show_value=True)
    try_n
    return (try_n,)


@app.cell(hide_code=True)
def box_answer():
    try_answer = mo.ui.text(value="", label="Your answer", full_width=True)
    try_answer
    return (try_answer,)


@app.cell(hide_code=True)
def box_cell(try_answer, try_n):
    mo.Html(C.box_picture(L.lang, try_n.value, try_answer.value))
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.md(r"""
    ### Exercise 4. Why the best?

    For each question, SQuAD takes the best score over its reference answers, and not their mean, nor the score against the first of them.
    """)
    return


@app.cell(hide_code=True)
def choice_best_cell():
    why_best = L.choice(
        "why_best",
        question="Why does SQuAD take the best score over the reference answers of a question?",
        options={
            "fast": "The best is quicker to work out than the mean of all of them",
            "right": "People mark different spans for one answer, and an answer matching any of them does as well as a person",
            "long": "So that a longer answer, which holds more of the words of the people, always scores higher than a short one",
            "first": "The first reference answer is the one that counts, and the others only break a tie",
        },
        correct="right",
        why={
            "right": "People who read the same paragraph mark *2018* or *in 2018*, and both are right. Against the first of the other answers only, people would score 68.1% exact match against each other instead of 82.4%, and 84.6% F1 instead of 91.8%. The best over all of them asks an answer to match the answer of one person, not those of all. Only different answers widen the choice: a copy of an answer changes no best score. 554 of the 1,190 English questions and 643 of the 1,183 Dutch ones have only one answer text, and there an answer that is right in other words scores low.",
            "fast": "Both are quick to work out. What do the reference answers of one question have in common?",
            "long": "An answer with too many words loses precision. Does the best over the reference answers favor long answers?",
            "first": "All reference answers count the same. What happens to an answer that matches the second one exactly?",
        },
    )
    why_best
    return (why_best,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Finding the answer in a passage](/course/open/nlp-extractive-qa?lang=en) your reader picked a sentence; here you scored answers the way SQuAD does, with the precision, recall and F1 score of [Precision and recall in one number: the F1 score](/course/open/ml-f1-score?lang=en) over words instead of examples, after the normalizing of [Normalizing: case, accents and stop words](/course/open/nlp-normalize?lang=en). People agree on 82.4% of the answers exactly and score 91.8% F1 against each other; a whole sentence holds the answer but scores low. The next lesson, *Translating by rules*, starts on machine translation, and later *Scoring a translation: BLEU and chrF* scores translations against reference translations in a similar way: with shared words, precision and recall.

    **An everyday example.** A language app that asks you to translate a sentence often accepts several translations, and marks your answer right when it matches one of them: the best over its reference answers.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_match, result_f1, result_scores, why_best):
    L.wrap_up(result_match, result_f1, result_scores, why_best)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
