# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-decontamination. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-decontamination", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Will it work on new data? Training and test sets](/course/open/ml-train-test-split?lang=en) and [Removing duplicates with MinHash](/course/open/llm-dedup?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_test():
    quick_test = L.choice(
        "quick-test",
        question="Why does a test set hold only examples that the model did not learn from?",
        options={
            "fast": "So that training takes less time: fewer examples to learn from",
            "hard": "So that the test is harder, as a fair score is a low one",
            "right": "So that its score says how well the model does on new examples",
        },
        correct="right",
        why={
            "fast": "Fewer examples to learn from is a side effect. The point is what the score measures: see [Will it work on new data? Training and test sets](/course/open/ml-train-test-split?lang=en).",
            "hard": "A test set is not meant to be harder, only new to the model. See [Will it work on new data? Training and test sets](/course/open/ml-train-test-split?lang=en).",
            "right": "And when test examples slip into the training data after all, the score no longer measures that. This lesson finds them.",
        },
    )
    quick_test
    return (quick_test,)


@app.cell(hide_code=True)
def quick_check_copy():
    quick_copy = L.choice(
        "quick-copy",
        question="A test question is on a web page, copied with one name changed. Which check finds it?",
        options={
            "both": "Both a check for exact copies and the overlap of their shingles",
            "right": "Only the overlap of their shingles: most runs of words are the same",
            "none": "Neither: one changed name makes it another text",
        },
        correct="right",
        why={
            "both": "One changed name makes the texts differ, so a check for exact copies misses it. See [Removing duplicates with MinHash](/course/open/llm-dedup?lang=en).",
            "none": "The runs of words away from the name stay the same, so the two texts still share almost all their shingles. See [Removing duplicates with MinHash](/course/open/llm-dedup?lang=en).",
            "right": "This lesson uses the same idea to look for test questions in training data, with longer runs of words.",
        },
    )
    quick_copy
    return (quick_copy,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A benchmark is only worth something when the model has not seen its questions. But a pretraining corpus is a large part of the web, and benchmarks are on the web: their files, pages that discuss them, and the texts they were made from. The makers of GPT-3 searched their training data for every overlap with the test sets of their benchmarks, and tried to remove it. A bug in the filter left some in, and training again cost too much. So they made a clean version of every benchmark, without the questions that might have leaked, and compared the scores. For the four Wikipedia benchmarks of language modelling and the Children's Book Test, almost all the text was in the training data, and they did not report those scores at all (Brown and others, 2020).

    **An everyday example.** A teacher who makes a test from the exercises the class did at home learns little from the scores: a high mark may mean that a pupil remembered the answers, not that they can do the sums.
    """)
    return


@app.cell(hide_code=True)
def predict_seen():
    _g = C.MEASURED["gpt"]
    guess_seen = L.guess(
        "seen",
        question=f"The course's GPT on tokens of [Project: your GPT on your own tokens](/course/open/gpt-train-bpe?lang=en) trained 10 times on the first 90% of 2,519 stories of Cosmopedia; the rest was its validation set. The course made {_g['of']} story-cloze questions from stories it trained on and {_g['of']} from stories it never saw: the start of a story, and four sentences, of which one really comes next. On which questions do you think it does better?",
        options={
            "seen": "Far better on the stories it trained on",
            "right": "About the same on both",
            "new": "Far better on the new stories",
        },
        reveal=f"**About the same: {_g['seen']} of {_g['of']} right on the stories it trained on, {_g['items']} of {_g['of']} on the new ones** (guessing gets about 1 in 4 right). A model of 1.3 million parameters that trained on 4.5 MB of stories remembers next to nothing of them word for word. Large models remember much more: the more parameters a model has, and the more often a text was in its training data, the more of it the model can repeat (Carlini and others, 2023). The course let its GPT pick the sentence whose probability rises most when the start of the story comes before it; a later lesson shows how a model scores options.",
    )
    guess_seen
    return (guess_seen,)


@app.cell(hide_code=True)
def text_idea():
    _m = C.MEASURED
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Contamination** is when test questions, or their answers, are in a model's training data: then a test score measures memory, not skill. It can be a whole question with its answer, or only the text a question was made from. That is a matter of degree: GPT-3's check flagged more than 90% of the questions of three reading tests, but in every overlap its makers looked at, only the text the questions ask about was in the training data, not the questions and answers.

        **Finding it with n-grams.** Cut every test question into runs of $n$ words, as the shingles of [Removing duplicates with MinHash](/course/open/llm-dedup?lang=en), in lower case, but now also without the punctuation at both ends of every word, so that a question copied in another layout still has the same runs. A question shorter than $n$ words counts as one whole. Then flag a question when more than a share of its n-grams occur in the training data. GPT-3 flagged a question when one of its 13-grams did, or, for benchmarks with short questions, one of its shorter runs.

        **The size matters.** A small $n$ flags common phrases: "once upon a time in a small town" is in many stories, and says nothing about a leak. A large $n$ misses a copy in which a word changed every few words. No n-gram check finds a question that was rewritten in other words, or translated.

        **Before and after training.** Before training, remove every document of the training data that shares an n-gram with a test question: **decontamination**. After training, when training again costs too much, flag the questions that share n-grams with the training data, and report the score on the **clean subset**, the questions that are not flagged, next to the score on all of them. When the score on the clean subset is much lower, the flagged questions may have lifted the score.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **contamination** | | when test questions, or their answers, are in a model's training data, so that a test score measures memory instead of skill | |
        | **decontamination** | | removing from the training data, before training, everything that shares an n-gram with a test question | |
        | **clean subset** | | the questions of a benchmark that share no n-gram with the training data; its score, next to the score on all questions, shows what contamination did | `report(flags, right)` |
        | **strip characters** | | `w.strip(chars)` takes every character of `chars` off both ends of `w`, as `strip()` takes off spaces | `"home!".strip(".!")` gives `"home"` |
        """),
        mo.md(r"""
        The picture: the course's 500 story-cloze questions of stories no model of the course has seen, against the 6,189 stories of TinyStories of the course. The two were made separately, by different models, and no question was copied into TinyStories.
        """),
        mo.Html(C.flagged_picture([(f"n = {_n}", _m["natural"][_n]) for _n in C.SIZES],
                                  {"title": "Flagged questions, by n", "of": "of {total}", "lang": L.lang,
                                   "note": "flagged: one n-gram is also in TinyStories",
                                   "alt": f"Bars of flagged questions: with n = 5, {_m['natural'][5]} of 500; with n = 8, {_m['natural'][8]}; with n = 13, {_m['natural'][13]}. A question is flagged when one of its n-grams is in TinyStories."})),
        mo.md(r"""
        With $n = 5$, 423 of the 500 questions look contaminated; about half of them start with "once upon a time, in", as 638 of the stories of TinyStories do. With $n = 13$, none do: these false positives are gone.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To keep a benchmark's questions out of a model's training data:

    1. Cut every test question into n-grams of its words: in lower case, without punctuation at their ends; a question shorter than $n$ words as one whole.
    2. Before training: remove every document of the training data that shares an n-gram with a test question.
    3. After training: flag the questions with more than a share of their n-grams in the training data, and report the score on all questions and on the clean subset.
    4. Look at what is flagged: a common phrase is not a leak, a copied question is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 1 and the check of step 3 for one short question and one document, with runs of 4 words, by hand.
    """)
    return


@app.cell
def example_overlap():
    _question = "Where did Tom hide the old key? Under the red flower pot."
    _document = "Tom looked around. He hid the old key under the red flower pot, by the door!"
    _marks = ".,!?"
    _sets = []
    for _text in (_question, _document):
        # 1. the words in lower case, without punctuation at their ends
        _words = []
        for _w in _text.lower().split():
            _words.append(_w.strip(_marks))
        # every run of 4 words, as a set
        _runs = set()
        for _i in range(len(_words) - 3):
            _runs.add(" ".join(_words[_i:_i + 4]))
        _sets.append(_runs)
    _shared = _sets[0] & _sets[1]
    print(len(_shared), "of the question's", len(_sets[0]), "runs are in the document:", sorted(_shared))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The question has 9 runs of 4 words, and 5 of them are in the document, from "the old key under" to "the red flower pot": without its punctuation, "key? Under" in the question reads as "key under", as in the document. The other 4 runs hold "hide", where the document says "hid": one changed word takes away up to 4 runs. Without `strip`, "key?" and "pot." would be other words than "key" and "pot,", and only 1 run would match. The share is 5 of 9, about 0.56, so this question is flagged for every `share` below that. With $n = 13$, it is too short for one 13-gram, and counts as one whole, which the document does not hold.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ngrams():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. N-grams of a question

        Complete `ngrams(text, n)`: step 1 of the recipe. Take the words of `text` in lower case, strip the characters of `PUNCT` off both ends of every word, and keep a word only when something is left. A text with fewer than `n` such words gives the whole text as one n-gram, its words with spaces between them, or no n-grams when it has no words. Otherwise, it gives the set of every run of `n` words in a row, each as one text with a space between the words. The check calls `ngrams("The dog barked at the post-man, \"Go home!\"", 4)`, and others.
        """),
        L.task("Make `ngrams` give every run of `n` cleaned words, and a short text as one whole."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ngrams():
    box_ngrams = L.code_box("ngrams", lines=16)
    box_ngrams
    return (box_ngrams,)


@app.cell(hide_code=True)
def check_ngrams(box_ngrams):
    result_ngrams = L.check("ngrams", box_ngrams.value)
    result_ngrams
    return (result_ngrams,)


@app.cell(hide_code=True)
def hints_ngrams():
    L.hints(
        "ngrams",
        [
            "The worked example does the first part and the runs. A word such as a lone dash becomes empty after `strip`: append it only `if w:`.",
            "Before the runs, check `len(words) < n`. Then the whole text is one n-gram, `{\" \".join(words)}`, unless `words` is empty.",
        ],
        why="Without capitals and punctuation, a question copied with another layout has the same n-grams. A short question has no run of n words, so without the whole text as one n-gram it could never be flagged; GPT-3 did the same.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_flag():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Flag the questions

        Complete `flag(test_grams, train_grams, share)`: the check of step 3. `test_grams` is a list with the set of n-grams of every test question, and `train_grams` the set of all n-grams of the training data. It gives back a list with `True` or `False` for every question: `True` when more than `share` of the question's n-grams are in `train_grams`, and `False` for a question without n-grams. With `share` 0, one n-gram is enough, as with GPT-3. The check calls `flag([{"a b", "b c", "c d"}, {"x y"}, set()], {"b c", "q r"}, 0)`, and others.
        """),
        L.task("Make `flag` mark every question with more than `share` of its n-grams in the training data."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flag():
    box_flag = L.code_box("flag", lines=12)
    box_flag
    return (box_flag,)


@app.cell(hide_code=True)
def check_flag(box_flag):
    result_flag = L.check("flag", box_flag.value)
    result_flag
    return (result_flag,)


@app.cell(hide_code=True)
def hints_flag():
    L.hints(
        "flag",
        [
            "`grams & train_grams` is the set of the question's n-grams that are in the training data. Its length, divided by `len(grams)`, is the share.",
            "Check `if not grams:` first: a question without n-grams would divide by 0.",
        ],
        why="A share lets you choose how much overlap counts: GPT-3 flagged on one shared n-gram, which flags many questions that only share a phrase; a higher share flags only questions that are mostly in the training data.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_report():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The score on the clean subset

        Complete `report(flags, right)`: the report of step 3. `flags` says for every question whether it is flagged, `right` whether the model got it right. It gives back three numbers: the share right of all questions; the share right of the clean subset, the questions that are not flagged, or `None` when there are none; and the share of questions that is flagged. The check calls `report` with 8 questions, of which 2 are flagged, and others.
        """),
        L.task("Make `report` give the score on all questions, on the clean subset, and the share flagged."),
    ])
    return


@app.cell(hide_code=True)
def exercise_report():
    box_report = L.code_box("report", lines=12)
    box_report
    return (box_report,)


@app.cell(hide_code=True)
def check_report(box_report):
    result_report = L.check("report", box_report.value)
    result_report
    return (result_report,)


@app.cell(hide_code=True)
def hints_report():
    L.hints(
        "report",
        [
            "Collect `right[i]` for every `i` where `flags[i]` is `False`: that is the clean subset.",
            "`sum` counts `True` as 1, so `sum(clean) / len(clean)` is the share right; check `if clean:` first.",
        ],
        why="GPT-3's makers did exactly this for every benchmark: when the score on the clean subset is about the same as on all questions, the leak did not lift the score much; when it is lower, it may have, or the clean questions are simply harder.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### What your functions find

    The demo runs your three functions on the course's story-cloze questions, the start of every story with the sentence that really follows. First against the stories of TinyStories of the course, for n = 5, 8 and 13. Then against a crawl that the course contaminates on purpose: the {C.number(len(C.stories()), 0, L.lang)} stories of TinyStories, with the stories of {C.PLANTED} of the questions copied in between, as a page that copied them would. A look-up model, as in [Will it work on new data? Training and test sets](/course/open/ml-train-test-split?lang=en), learns that crawl: it remembers every text, looks for the last {C.KEY} words of a question's start, and picks the sentence that follows there, or guesses when it finds nothing. It takes about ten seconds.
    """)
    return


@app.cell(hide_code=True)
def demo_natural(box_ngrams, box_flag, result_ngrams, result_flag):
    mo.stop(not (result_ngrams.state == "correct" and result_flag.state == "correct"),
            mo.md("*Finish exercises 1 and 2 first: this part runs your `ngrams` and your `flag`.*"))
    _out = C.natural(run_code(box_ngrams.value).names["ngrams"], run_code(box_flag.value).names["flag"])
    _rows = []
    for _n in C.SIZES:
        _ex = _out[_n]["example"]
        _rows.append(f"| {_n} | {C.number(_out[_n]['flagged'], 0, L.lang)} | {'none' if _ex is None else '“' + _ex + '”'} |")
    mo.md("Your functions on the questions against TinyStories, with `share` 0:\n\n| n | questions flagged, of 500 | the n-gram they share most often |\n|---|---|---|\n" + "\n".join(_rows) +
          "\n\nNo question was copied into TinyStories, so every flag here is a false positive: a phrase that many stories use.")
    return


@app.cell(hide_code=True)
def demo_crawl(box_ngrams, box_flag, box_report, result_ngrams, result_flag, result_report):
    _ready = result_ngrams.state == "correct" and result_flag.state == "correct" and result_report.state == "correct"
    mo.stop(not _ready, mo.md("*Finish exercises 1 to 3 first: this part runs your three functions.*"))
    _fns = {_name: run_code(_box.value).names[_name] for _box, _name in ((box_ngrams, "ngrams"), (box_flag, "flag"), (box_report, "report"))}
    _r = C.contaminated(_fns["ngrams"], _fns["flag"], _fns["report"])
    _score, _clean, _share = _r["report"]
    _removed = "; ".join(f"n = {_n}: {C.number(_r['removed'][_n], 0, L.lang)}" for _n in C.SIZES)
    mo.md(f"""With n = {C.N}, your `flag` marks {_r['flagged']} questions: the {_r['planted_flagged']} that were copied, and {_r['flagged'] - _r['planted_flagged']} others, which share a run of 13 words with a copied one, mostly at the start of the story. Cosmopedia's stories often start the same way, such as "once upon a time in a small town named Harmonyville lived two best friends".

The look-up model that learned the crawl gets {C.number(100 * _score, 1, L.lang)}% of all questions right, and {C.number(100 * _clean, 1, L.lang)}% of the clean subset: about 1 in 4, what guessing gets. It knows the copied questions by heart, and nothing else, and that lifts its score on all questions. Your `report` says {C.number(100 * _share, 1, L.lang)}% of the questions are flagged.

Decontaminating the crawl, by removing every document that shares an n-gram with a question, removes this many of its {C.number(_r['docs'], 0, L.lang)} documents: {_removed}. With n = 13, that is exactly the {_r['planted']} copies; with n = 5, most of TinyStories goes, for phrases such as "once upon a time there". After decontamination with n = 13, the look-up model gets {C.number(100 * _r['after'], 1, L.lang)}% right: its score now measures what it can do.""")
    return


@app.cell(hide_code=True)
def text_exercise_after():
    mo.md(r"""
    ### Exercise 4. Found after training
    """)
    return


@app.cell(hide_code=True)
def choice_after_cell():
    choice_after = L.choice(
        "after-training",
        question="After training, you find that part of a benchmark was in the training data, and training again costs too much. What can you still do?",
        options={
            "remove": "Remove those questions from the training data now",
            "nothing": "Nothing: the model's score on that benchmark can no longer be trusted at all",
            "right": "Report the score on the clean subset next to the score on all questions",
        },
        correct="right",
        why={
            "remove": "The model has already learned from them: removing them now changes nothing in its parameters. That only helps before training.",
            "nothing": "The clean subset still gives a score on questions that did not leak, and comparing it with the score on all questions shows whether the leak may have lifted it.",
            "right": "That is what GPT-3's makers did. Where the two differed, they looked closer, and marked two results, PIQA and Winograd, for possible contamination. Where almost a whole benchmark was in the training data, they reported no score at all.",
        },
    )
    choice_after
    return (choice_after,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The test set of [Will it work on new data? Training and test sets](/course/open/ml-train-test-split?lang=en), now kept apart from a whole corpus; the shingles of [Removing duplicates with MinHash](/course/open/llm-dedup?lang=en), now as n-grams of test questions; the question of [Reading AI claims critically](/course/open/res-reading-claims?lang=en) whether a benchmark was in the training data, now something you can check. The next lesson turns the cleaned stories into a file of token ids to train on.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_ngrams, result_flag, result_report, choice_after):
    L.wrap_up(result_ngrams, result_flag, result_report, choice_after)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
