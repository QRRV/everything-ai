# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-bert. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-bert", "en")
    C = L.checks   # BERT-Tiny's scores at the course's blanks, the ranks measured on Tatoeba, BERT-Tiny, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions, about [Encoders, decoders and the GPT line](/course/open/lm-model-families?lang=en) and [One vector per word, or one per use](/course/open/nlp-static-vs-contextual?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_encoder():
    quick_encoder = L.choice(
        "quick-encoder",
        question="In *Encoders, decoders and the GPT line*, a text of six tokens goes into an encoder. Which tokens does the third token look at?",
        options={
            "before": "The first two, and itself",
            "right": "All six",
            "after": "The three after it, and itself",
        },
        correct="right",
        why={
            "right": "In an encoder, every token looks at all tokens, before and after it. That makes the way a decoder learns, guessing the next token, far too easy: this lesson shows why, and what BERT learns from instead.",
            "before": "That is a decoder, such as GPT: every token looks at itself and the tokens before it. Which tokens does a decoder hide, and an encoder not? See [Encoders, decoders and the GPT line](/course/open/lm-model-families?lang=en).",
            "after": "None of the models in *Encoders, decoders and the GPT line* looks only ahead. Does an encoder hide the tokens before the third one? See [Encoders, decoders and the GPT line](/course/open/lm-model-families?lang=en).",
        },
    )
    quick_encoder
    return (quick_encoder,)


@app.cell(hide_code=True)
def quick_check_pieces():
    quick_pieces = L.choice(
        "quick-pieces",
        question="In *One vector per word, or one per use*, what does `bert.pieces(\"kayak\")` give?",
        options={
            "whole": "`['[CLS]', 'kayak', '[SEP]']`",
            "bare": "`['kay', '##ak']`",
            "right": "`['[CLS]', 'kay', '##ak', '[SEP]']`",
            "letters": "`['[CLS]', 'k', '##a', '##y', '##a', '##k', '[SEP]']`",
        },
        correct="right",
        why={
            "right": "BERT-Tiny cuts *kayak* into *kay* and *##ak*, where *##* means that the piece continues a word, and puts *[CLS]* before them and *[SEP]* after them. In this lesson BERT-Tiny guesses pieces like these.",
            "whole": "*kayak* is not one of the 30,522 pieces of BERT-Tiny. What does `bert.pieces` do with a word it does not have? See [One vector per word, or one per use](/course/open/nlp-static-vs-contextual?lang=en).",
            "bare": "`bert.pieces` gives the pieces the model reads, and the model reads more than the word. What else goes into BERT-Tiny with every text? See [One vector per word, or one per use](/course/open/nlp-static-vs-contextual?lang=en).",
            "letters": "BERT-Tiny uses single letters only where nothing longer fits. Which longer pieces could make up *kayak*? See [One vector per word, or one per use](/course/open/nlp-static-vs-contextual?lang=en).",
        },
    )
    quick_pieces
    return (quick_pieces,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A decoder learns by guessing the next token: every token looks only at the tokens before it, so the next one stays hidden. In an encoder, every token looks at all tokens, the next one too. Guessing the next token then teaches nothing: the answer is already in the input. Jacob Devlin and colleagues at Google, who put BERT online in 2018 and published it in 2019, wrote that looking both ways "would allow each word to indirectly 'see itself', and the model could trivially predict the target word in a multi-layered context."

    Their way out was to hide some words of a text, and let the model guess them from the words on both sides. In this lesson you build a training example of that kind, measure how well BERT-Tiny guesses, and compute the loss it learned from.

    **An everyday example.** In *The ___ was barking all night, so nobody slept.*, the words after the gap settle what is missing: something that barks.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_first = L.guess(
        "guess",
        question="The course took 500 English sentences of Tatoeba, a collection of sentences that volunteers write and translate, and hid every word that BERT-Tiny keeps as one piece, one word at a time: 3,031 words in all. For how many of them is BERT-Tiny's first guess, the piece with the highest score, the hidden word?",
        options={
            "low": "About 3 in 100",
            "right": "About 30 in 100",
            "mid": "About 60 in 100",
            "high": "Nearly always: 90 in 100",
        },
        reveal="**About 30 in 100**: 919 of the 3,031, 0.303. The hidden word is among the first five guesses for 1,634 of them, 0.539. Many misses are good guesses all the same: in *I drink a cup of [MASK] every morning.*, BERT-Tiny's first guess is *breakfast*, and *coffee* comes second.",
    )
    guess_first
    return (guess_first,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Masked language modeling

    **Masked language modeling** (MLM) trains a model by hiding some pieces of a text and letting it guess them from the rest. Daniel Jurafsky and James H. Martin write in *Speech and Language Processing*: "instead of predicting the following word, we mask a word in the middle and ask the model to guess the word given the words on both sides." Such a fill-in-the-blank test is older than computers: it is the **cloze task**, which W. L. Taylor described in 1953 as a way to measure how readable a text is.

    BERT chooses 15% of the pieces of a text at random. A chosen piece becomes **[MASK]** 80% of the time, a random piece 10% of the time, and stays as it is 10% of the time. The paper's example is *my dog is hairy*, with *hairy* chosen: BERT then sees *my dog is [MASK]*, *my dog is apple* or *my dog is hairy*. At every chosen place, the model has to give the original piece. The other places help it guess, but do not count in the loss. Why not [MASK] at every chosen place? That is exercise 2.

    ### From a vector to a guess

    In BERT-Tiny, every place has a vector of 128 numbers after the last block. The **head** turns it into a guess. It passes the vector through one more small layer, then takes its dot product with the vector of every piece in the model's table of 30,522 pieces, the same table the model starts from, and adds one number per piece: 30,522 scores. Using one table at both ends is called weight tying; a later lesson, *Weight tying*, goes into it. Softmax turns the scores into probabilities, as in [Many classes: softmax](/course/open/ml-softmax?lang=en). The loss at a chosen place is the cross-entropy, as in [Cross-entropy loss for many classes](/course/open/ml-cross-entropy?lang=en): minus the log of the probability of the right piece. The loss of a text is the mean over its chosen places.

    The **rank** of a piece is its place in the list of all pieces, ranked from the highest score down: 1 for the first guess.

    ### What the words after the gap are worth

    BERT-Tiny learned with both sides. To see what the words after a gap give it, the course hid the 3,031 words of Tatoeba once more, and this time also made every piece after the gap a [MASK], so that only the words before it are left. That asks a lot of BERT-Tiny: it never learned with so many [MASK]s in a row, while a decoder learns with only the words before. The picture shows the difference all the same.
    """)
    return


@app.cell(hide_code=True)
def idea_sides():
    mo.Html(C.sides_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.vstack([
        mo.md(r"""
        With both sides, BERT-Tiny's first guess is the hidden word for 0.303 of the 3,031; with only the words before, for 0.140, less than half as often. Half of the hidden words have rank 4 or better with both sides, and rank 25 or better with only the words before.

        ### Next sentence prediction

        BERT learned a second task at the same time: **next sentence prediction**. It got two sentences, and had to say whether the second came right after the first in the text they were taken from, or from somewhere else; half of the pairs were of each kind. BERT's authors found that leaving this task out hurt three of the tasks they measured. In 2019, Yinhan Liu and colleagues trained BERT again, as RoBERTa, and found that leaving it out "matches or slightly improves downstream task performance, in contrast to Devlin et al. (2019)." DistilBERT and ModernBERT leave it out too.

        ### The BERT family

        | Model | Year | Parameters | What changed |
        |---|---|---|---|
        | BERT | 2019 | 110 million, 340 million | MLM with 15% of the pieces, plus next sentence prediction; 3.3 billion words of books and English Wikipedia |
        | BERT-Tiny | 2019 | 4.4 million | the same recipe, with 2 blocks and vectors of 128 numbers; released in 2020 |
        | RoBERTa | 2019 | 355 million | trained longer, with bigger batches, on 160 GB of text instead of 16 GB; no next sentence prediction |
        | DistilBERT | 2019 | 66 million | half the blocks, trained to give the same answers as BERT |
        | DeBERTa | 2020 | 134 million, 1.5 billion | one vector for a piece and one for its place, kept apart in attention |
        | ModernBERT | 2024 | 149 million, 395 million | texts of up to 8,192 tokens; 2 trillion tokens, code among them; 30% of the pieces hidden |
        | EuroBERT | 2025 | 210 million, 610 million, 2.1 billion | 15 languages, Dutch among them; up to 8,192 tokens; 50% of the pieces hidden, 10% in the last stage |

        Benjamin Warner and colleagues, who made ModernBERT, explain why such models are still made: "Encoder-only transformer models such as BERT offer a great performance-size tradeoff for retrieval and classification tasks with respect to larger decoder-only models."

        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **masked language modeling** (MLM) | training a model by hiding some pieces of a text and letting it guess them from the pieces on both sides |
        | **cloze task** | a fill-in-the-blank test: a text with gaps, to be filled in from the words around them |
        | **BERT** | an encoder from Google (2018, published 2019) that learned with MLM; the name stands for Bidirectional Encoder Representations from Transformers |
        | **[MASK]** | the piece that stands in for a hidden piece |
        | **head** | the last part of the model: from the vector of a place to a score for every piece |
        | **rank** | the place of a piece in the list of all pieces, ranked from the highest score down; 1 for the first guess |
        | **next sentence prediction** | BERT's second task: does the second of two sentences come right after the first? |
        | `t[[2, 5]]` | rows 2 and 5 of the tensor or array `t`, in that order, as a new table of two rows |
        | `t[rows]` | the same with a list of row numbers `rows`, such as `rows = [2, 5]` |
        """),
        L.watch_out("The rank and the loss look at one hidden piece only. *breakfast* in *I drink a cup of [MASK] every morning.* fits the sentence well, but counts as a miss, because the course hid *coffee*."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Cut the text into pieces, and choose 15% of the places at random.
    2. At each chosen place, put [MASK] 80% of the time, a random piece 10% of the time, and keep the piece 10% of the time. The label of a chosen place is its original piece; every other place gets no label.
    3. Run the model, and let the head give a score for every piece at every place.
    4. At each chosen place, take the cross-entropy of its scores with the right piece, and average over the chosen places.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The button below downloads BERT-Tiny. The code then looks at the [MASK] of *I drink a cup of [MASK] every morning.* The course's `bert.piece_scores` gives the scores of BERT-Tiny's head at every place of a text: one row of 30,522 scores per piece. `bert.tokenizer().vocab` is the list of BERT-Tiny's 30,522 pieces, so that `vocab[j]` is the piece with number `j`. The code turns the row at the [MASK] into probabilities with softmax, and prints the best guess, then the probability and the loss of *coffee*.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download BERT-Tiny, 18 MB")
    return (get_button,)


@app.cell(hide_code=True)
def bert_cell(get_button):
    bert = C.bert_or_stop(get_button, {
        "ask": "*BERT-Tiny is not on your computer yet. The button downloads it once from Hugging Face, at a fixed version: `google/bert_uncased_L-2_H-128_A-2`, 18 MB, under the Apache-2.0 license. The worked example and the part where you try it yourself need it; the exercises do not.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: a file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is not enough room on your disk. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    mo.md("*BERT-Tiny is downloaded.*")
    return (bert,)


@app.cell
def example_cell(bert):
    import math
    import numpy as np

    sentence = "I drink a cup of [MASK] every morning."
    pieces = bert.pieces(sentence)
    k = pieces.index("[MASK]")
    scores = bert.piece_scores(sentence)[k]
    p = np.exp(scores) / np.sum(np.exp(scores))
    vocab = bert.tokenizer().vocab
    print(pieces)
    print(k, len(scores))
    best = np.argmax(p)
    print(vocab[best], round(float(p[best]), 3))
    coffee = vocab.index("coffee")
    print("coffee", round(float(p[coffee]), 3), round(-math.log(p[coffee]), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The sentence has 11 pieces, and the [MASK] stands at place 6, counted from 0, with *[CLS]* at place 0. The row has 30,522 scores, one per piece. The best guess is *breakfast*, with a probability of 0.069. *coffee* gets 0.061, so it comes second, and its loss, minus the log of its probability, is 2.803.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_example():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. make_example: a training example

        Write `make_example(pieces, replace)`. `pieces` is a list of pieces; `replace` is a plan: a dict with a chosen place as key and what goes in there as value, `"[MASK]"`, a random piece, or the piece itself. Give back two lists, each as long as `pieces`: `inputs`, what the model sees at every place, and `labels`, the piece the model has to give at every chosen place, and None at every other place.

        For `make_example(["the", "cat", "sat", "on", "the", "mat", "."], {1: "[MASK]", 5: "[MASK]"})` the result is `(["the", "[MASK]", "sat", "on", "the", "[MASK]", "."], [None, "cat", None, None, None, "mat", None])`. Replace the `...`. The check calls that and `make_example(["i", "drink", "a", "cup", "of", "coffee", "."], {1: "[MASK]", 3: "river", 5: "coffee"})`, with a random piece at place 3 and the piece itself at place 5, then cases of its own: no plan, a plan for every place, a piece that occurs twice, a plan written out of order with a piece that continues a word, and the 18 pieces of a sentence of the course, `C.EXAMPLE_PIECES`, with the plan `C.EXAMPLE_PLAN`.
        """),
        L.task("Give back what the model sees at every place, and the piece it has to give at every chosen place, or None."),
    ])
    return


@app.cell(hide_code=True)
def exercise_example():
    box_example = L.code_box("make_example", lines=11)
    box_example
    return (box_example,)


@app.cell(hide_code=True)
def check_example(box_example):
    result_example = L.check("make_example", box_example.value)
    result_example
    return (result_example,)


@app.cell(hide_code=True)
def hints_example():
    L.hints(
        "make_example",
        [
            "Two `append`s for every place `k`: one to `inputs`, one to `labels`. Use `if k in replace:` for a chosen place, and `else:` for every other place. At a chosen place, the plan says what goes into `inputs`.",
            "A label is the answer the model has to give at that place. It is the same whether the plan puts in `\"[MASK]\"`, a random piece or the piece itself: the model has to undo the plan. Which of the two arguments still knows what stood there?",
            "The line under `if k in replace:` that fills `labels`: `            labels.append(___)`",
        ],
        why=r"`labels.append(pieces[k])`: the label is the piece that stood at place `k`, whatever the plan puts there. In full, under `for k in range(len(pieces)):`: `if k in replace:` with `inputs.append(replace[k])` and `labels.append(pieces[k])` under it; then `else:` with `inputs.append(pieces[k])` and `labels.append(None)`. In the example, places 1 and 5 get `\"[MASK]\"` in `inputs`, and *cat* and *mat* in `labels`.",
    )
    return


@app.cell(hide_code=True)
def text_after_example_exercise():
    mo.md(r"""
    In the picture, your function made the training example of a sentence of the course. BERT makes a plan like this one for every text: 15% of the places at random, and for each of them [MASK], a random piece or the piece itself. According to RoBERTa's authors, the original BERT drew its plans once, before training, and copied its data 10 times, so that a text came with 10 different plans over the 40 times it went through the data. RoBERTa draws a new plan every time a text comes by, and its authors found that "dynamic masking is comparable or slightly better than static masking."
    """)
    return


@app.cell(hide_code=True)
def text_exercise_keep():
    mo.md(r"""
    ### Exercise 2. Why not always [MASK]?
    """)
    return


@app.cell(hide_code=True)
def choice_keep_cell():
    choice_keep = L.choice(
        "keep-some",
        question="At a chosen place, BERT puts [MASK] only 80% of the time; 10% of the time it puts a random piece there, and 10% of the time it keeps the piece. Why not [MASK] at every chosen place?",
        options={
            "speed": "With fewer [MASK]s, training goes faster",
            "right": "A text BERT reads later has no [MASK] in it",
            "typos": "The random pieces teach BERT to fix the typos in a text",
        },
        correct="right",
        why={
            "right": "A model like BERT is trained further and then used on ordinary texts, without [MASK]. The paper calls that \"a mismatch between pre-training and fine-tuning, since the [MASK] token does not appear during fine-tuning\"; fine-tuning is training a model further for one task, the subject of a later lesson, *Fine-tuning an encoder for a task*. With random pieces and kept pieces in the mix, the model cannot tell which places it will be asked about, so, in the paper's words, it is \"forced to keep a distributional contextual representation\" of every piece it reads.",
            "speed": "A step costs the same either way: the model computes a vector for every place, whatever stands there, and the loss counts the same chosen places. Think of the texts BERT gets once its training is done.",
            "typos": "BERT does learn to give the right piece where a random one stands, but random pieces are rare: 1.5% of all pieces, 10% of 15%. The paper notes that this \"does not seem to harm the model's language understanding capability.\" Think of the texts BERT gets once its training is done.",
        },
    )
    choice_keep
    return (choice_keep,)


@app.cell(hide_code=True)
def text_exercise_rank():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. rank: how far down is the right piece?

        Write `rank(scores, right)`. `scores` is a NumPy array with one score per piece, such as BERT-Tiny's 30,522 scores at a [MASK], and `right` is the number of the right piece. Give back its rank: its place in the list of all pieces, ranked from the highest score down, 1 for the piece with the highest score. A piece with the same score as the right piece does not stand above it.

        For `rank(np.array([0.5, 2.0, 1.0, -1.0]), 2)` the result is 2: 1.0 is the second-highest score. Replace the `...`. The check calls that and `rank(np.array([0.5, 2.0, 1.0, -1.0]), 1)`, which should give 1, then cases of its own: the lowest score, two pieces with the same score at the top and two lower down, only negative scores, and the scores of BERT-Tiny at the [MASK] of three sentences of the course, `C.blank_scores(1)`, `C.blank_scores(6)` and `C.blank_scores(7)`, with the number of the hidden word: *coffee*, *capital* and *rain*.
        """),
        L.task("Give back the rank of the piece with number `right`: 1 plus the number of pieces that stand above it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rank():
    box_rank = L.code_box("rank", lines=7)
    box_rank
    return (box_rank,)


@app.cell(hide_code=True)
def check_rank(box_rank):
    result_rank = L.check("rank", box_rank.value)
    result_rank
    return (result_rank,)


@app.cell(hide_code=True)
def hints_rank():
    L.hints(
        "rank",
        [
            "Two lines: `better = np.sum(...)`, the number of pieces that stand above the right piece, then `return better + 1`. Inside `np.sum` goes a comparison of the whole array `scores` at once, as with a boolean mask.",
            "A comparison of an array with one number, such as `scores > 0`, gives an array of `True` and `False`, one per piece, and `np.sum` counts the `True`s. With which number do you compare, so that exactly the pieces above the right piece give `True`? And which comparison leaves out a piece with the same score?",
            "The first line: `    better = np.sum(___)`",
        ],
        why=r"`better = np.sum(scores > scores[right])`: `scores[right]` is the score of the right piece, and `>` gives `True` only for a higher score, so a piece with the same score does not count, and neither does the right piece itself. Then `return better + 1`. In the example, only 2.0 is higher than 1.0, so the rank is 2.",
    )
    return


@app.cell(hide_code=True)
def text_after_rank():
    mo.md(r"""
    In the picture, your function ranked the hidden word of all eight sentences of the course. BERT-Tiny's first guess is right in 1 of the 8, and the hidden word is among its first five guesses in 4. Many misses are words of the right kind: *breakfast* for *coffee*, *monday* for *sunday*, *hot* for *cold*. *rain* gets rank 341: the first guesses are *go*, *do*, *it*, *be* and *move*, and most of them fit the grammar of *It started to [MASK]*, but none fits the umbrellas. Victor Sanh and colleagues give the guesses of the version of BERT with 110 million parameters for *I think this is the beginning of a beautiful [MASK]*: "two high probability tokens (day and life) and a long tail of valid predictions (future, story, world. . . )." A rank of 3 can be a good answer too.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_loss():
    mo.vstack([
        mo.md(r"""
        ### Exercise 4. mlm_loss: the loss at the chosen places

        Write `mlm_loss(scores, places, right)`. `scores` is a tensor with one row of scores per place of a text, as the head gives it; `places` is a list of the chosen places, and `right` a list with the number of the right piece at each of them, in the same order. Give back the mean cross-entropy over the chosen places, and over no other place: one number, or a tensor with one number.

        For `mlm_loss(torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [1.0, 2.0, 0.5]]), [2], [0])` the result is 1.4644, rounded: minus the log of the probability that softmax gives piece 0 in row 2. Replace the `...`. The check calls that and the same table with `[0, 2]` and `[0, 1]`, which should give 0.352, then cases of its own: those two places in the other order, one place where the right piece has the highest score, every place, a table of 4 rows of 5 with three places, and BERT-Tiny's scores at every place of a sentence of the course with two blanks, `C.week_scores()`, with the places 5 and 11 and the numbers of their hidden words.
        """),
        L.task("Give back the mean cross-entropy over the chosen places only."),
    ])
    return


@app.cell(hide_code=True)
def exercise_loss():
    box_loss = L.code_box("mlm_loss", lines=7)
    box_loss
    return (box_loss,)


@app.cell(hide_code=True)
def check_loss(box_loss):
    result_loss = L.check("mlm_loss", box_loss.value)
    result_loss
    return (result_loss,)


@app.cell(hide_code=True)
def hints_loss():
    L.hints(
        "mlm_loss",
        [
            "One line: `return nn.CrossEntropyLoss()(...)`, as in [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en): first a table of scores with one row per prediction, then a tensor with the right class of each row. Here the right classes are `torch.tensor(right)`.",
            "`torch.tensor(right)` has one number per chosen place, so the table of scores needs one row per chosen place, in the same order, and no other rows. The words table shows how to take several rows of a tensor at once. Which list says which rows?",
            "The one line: `    return nn.CrossEntropyLoss()(___, torch.tensor(right))`",
        ],
        why=r"`return nn.CrossEntropyLoss()(scores[places], torch.tensor(right))`: `scores[places]` takes the rows of the chosen places, in the order of `places`, and `nn.CrossEntropyLoss()` gives the mean of minus the log of the probability of the right piece in each row. In the example, softmax of row 2, `[1.0, 2.0, 0.5]`, gives piece 0 a probability of 0.231, and minus its log is 1.4644.",
    )
    return


@app.cell(hide_code=True)
def text_after_loss():
    mo.md(r"""
    In the picture, your function computed the loss of *A week has seven [MASK] and a year has twelve [MASK].* At the two hidden places, BERT-Tiny's first guess is *years* both times, and the mean loss is 2.99. Where BERT-Tiny sees the piece, the loss is below 0.09: the answer is right there in its input. Over all 12 places the mean is 0.52, and almost all of it comes from the two hidden places. A loss over every place would mostly reward copying. Jurafsky and Martin note the price: "only 15% of the input samples in the training data are actually used for training weights." ELECTRA, a later model of the family, learns from every piece.
    """)
    return


@app.cell(hide_code=True)
def text_explore():
    mo.md(r"""
    ### Try it: fill in the blank

    Type an English sentence with one or more [MASK]s, in capitals and with the square brackets. The picture shows BERT-Tiny's five best pieces for each [MASK], with both sides and with only the words before. The sentence below gives *winter* first with both sides; with only *the* before the gap, BERT-Tiny can only guess common words. Try a gap whose answer comes from the words after it, and one whose answer the words before already give.
    """)
    return


@app.cell(hide_code=True)
def explore_controls():
    explore_text = mo.ui.text(value="The [MASK] is very cold in winter.", label="Sentence", full_width=True)
    explore_text
    return (explore_text,)


@app.cell(hide_code=True)
def explore_look(bert, explore_text):
    bert   # BERT-Tiny is downloaded: this cell waits for the button above
    mo.Html(C.explore_picture(explore_text.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Encoders, decoders and the GPT line](/course/open/lm-model-families?lang=en), every token of an encoder looked at all tokens. That makes guessing the next token too easy, so BERT learns by masked language modeling: it hides 15% of the pieces, and its head guesses each of them from both sides, with the cross-entropy at the chosen places only as its loss, as in [How good is a language model?](/course/open/lm-nll?lang=en). BERT-Tiny's first guess is the hidden word for 0.303 of the words of Tatoeba, and for 0.140 with only the words before. The next lesson of the domain, *Fine-tuning an encoder for a task*, trains such a model further for one task.

    **An everyday example.** Google Search has used BERT since October 2019, first for searches in English in the U.S. Pandu Nayak of Google wrote then that "BERT models can therefore consider the full context of a word by looking at the words that come before and after it—particularly useful for understanding the intent behind search queries." In *2019 brazil traveler to usa need a visa*, the small word *to* decides who travels where.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_keep, result_example, result_loss, result_rank):
    L.wrap_up(result_example, choice_keep, result_rank, result_loss)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
