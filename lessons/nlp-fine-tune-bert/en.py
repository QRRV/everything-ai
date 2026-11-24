# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-fine-tune-bert. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-fine-tune-bert", "en")
    C = L.checks   # BERT-Tiny's numbers for six sentences with a head, the scores of the course's fine-tuned BERT-Tiny for names, the demo, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions, about [Reusing a trained network: freeze, then fine-tune](/course/open/torch-fine-tune?lang=en) and [A label per word: sequence labeling and BIO](/course/open/nlp-bio-tags?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_phases():
    quick_phases = L.choice(
        "quick-phases",
        question="In *Reusing a trained network: freeze, then fine-tune*, what happens in phase 2?",
        options={
            "head": "Only the new last layer learns, with a larger learning rate",
            "right": "Every layer learns, with a learning rate ten times smaller",
            "scratch": "Every layer learns again, from random numbers",
            "swap": "The new last layer is frozen, and the rest learns",
        },
        correct="right",
        why={
            "right": "Phase 1 trains only the new last layer on the frozen features; phase 2 lets every layer learn, with steps ten times smaller, so that the borrowed features change only a little. In this lesson, the borrowed network is BERT-Tiny.",
            "head": "That is phase 1, and there the learning rate is the larger one. Which parameters learn in phase 2, and how big are their steps? See [Reusing a trained network: freeze, then fine-tune](/course/open/torch-fine-tune?lang=en).",
            "scratch": "Then the borrowed features would be thrown away, and the network would start over. What does phase 2 keep of the trained network? See [Reusing a trained network: freeze, then fine-tune](/course/open/torch-fine-tune?lang=en).",
            "swap": "The new last layer starts with random numbers, so it is the part that has the most to learn. Which parameters does phase 2 unfreeze? See [Reusing a trained network: freeze, then fine-tune](/course/open/torch-fine-tune?lang=en).",
        },
    )
    quick_phases
    return (quick_phases,)


@app.cell(hide_code=True)
def quick_check_bio():
    quick_bio = L.choice(
        "quick-bio",
        question="In *A label per word: sequence labeling and BIO*, which tag does the second word of the name of a person get?",
        options={
            "b": "B-PER again, since it is part of the name",
            "o": "O, since only the first word marks the name",
            "right": "I-PER: inside the name",
            "bare": "PER, the type alone",
        },
        correct="right",
        why={
            "right": "B starts a name, I continues it, and O is outside every name: *Ada Lovelace* is `B-PER I-PER`. In this lesson, BERT-Tiny learns these tags, for persons, places and organizations.",
            "b": "Then *Ada Lovelace* would be two names of one word each. Which letter says that a word continues the name before it? See [A label per word: sequence labeling and BIO](/course/open/nlp-bio-tags?lang=en).",
            "o": "Then a name of two words would look like a name of one word followed by an ordinary word. Which letter marks the words inside a name? See [A label per word: sequence labeling and BIO](/course/open/nlp-bio-tags?lang=en).",
            "bare": "With the type alone, two names right after each other would melt into one. Which letter does BIO put before the type of a word that continues a name? See [A label per word: sequence labeling and BIO](/course/open/nlp-bio-tags?lang=en).",
        },
    )
    quick_bio
    return (quick_bio,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    BERT-Tiny learned by filling in blanks: it never saw a review with the label *positive*, or a name with the tag of a person. Still, the vectors it gives after its last block are a good start for such tasks. Jacob Devlin and colleagues, who made BERT, describe how little a task adds: "For each task, we simply plug in the task-specific inputs and outputs into BERT and fine-tune all the parameters end-to-end." And: "Compared to pre-training, fine-tuning is relatively inexpensive."

    In this lesson you put two kinds of head on BERT-Tiny: one that labels a sentence, and one that tags every word. You fine-tune BERT-Tiny on the 2,400 review sentences of [Logistic regression on words](/course/open/nlp-logistic-text?lang=en), and you read the names that a BERT-Tiny fine-tuned by the course finds in the test sentences of Universal NER.

    **An everyday example.** Someone who reads English well learns in an afternoon to sort letters from customers into praise and complaints. Nobody teaches them English again first: they only learn what the new job asks.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_frozen = L.guess(
        "guess",
        question="In *Logistic regression on words*, a weight for every word labeled 491 of the 600 held-out review sentences right. Now BERT-Tiny stays frozen, and only a new layer on its vector at [CLS] learns, from the same 2,400 training sentences, for 4 epochs. How many of the 600 does it label right?",
        options={
            "coin": "About 300, no better than a coin toss",
            "right": "About 440, fewer than the words",
            "same": "About 490, as many as the words",
            "more": "About 560, far more than the words",
        },
        reveal="**About 440**: over 5 seeds, the course measured 430 to 449. Frozen, BERT-Tiny's vector at [CLS] tells the new layer less about praise and complaints than the words themselves told logistic regression. The demo below shows what happens when all of BERT-Tiny learns as well.",
    )
    guess_frozen
    return (guess_frozen,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### A head for the task

    BERT-Tiny gives a vector of 128 numbers at every place of a text, after its last block. In [BERT: filling in the blank](/course/open/nlp-bert?lang=en), a head turned the vector of a place into a score for every piece. For a new task, BERT-Tiny gets a new **head**: a layer that turns its vectors into scores for that task. **Fine-tuning** then trains on labeled examples of the task, as in *Reusing a trained network*: only the head, with BERT-Tiny frozen, or the head and all of BERT-Tiny, with a small learning rate.

    **To label a sentence**, the head reads one vector: the one at **[CLS]**, the piece that BERT puts before every text. `nn.Linear(128, 2)` turns it into 2 scores, negative and positive, and the loss is the cross-entropy, as always. Daniel Jurafsky and James H. Martin write in *Speech and Language Processing*: "The output vector in the final layer of the model for the [CLS] input represents the entire input sequence and serves as the input to a classifier head, a logistic regression or neural network classifier that makes the relevant decision." They name another way as well: "take the sum or the mean of the last output vector from each token in the sequence."

    ### Token classification

    **Token classification** puts a head of the same kind on every place: `nn.Linear(128, 7)` gives 7 scores at every piece, one for each tag of BIO with persons, places and organizations: `O`, `B-PER`, `I-PER`, `B-LOC`, `I-LOC`, `B-ORG` and `I-ORG`, numbered 0 to 6.

    The tags belong to words, and BERT-Tiny reads pieces: *Lovelace* is *love* and *##lace*. BERT's authors read a word at its first piece: "We use the representation of the first sub-token as the input to the token-level classifier over the NER label set." In this lesson, the first piece of every word gets the tag of its word in training, and every other place gets the label **-100**: [CLS], [SEP] and every piece that continues a word. `nn.CrossEntropyLoss` leaves a place with -100 out. The PyTorch documentation says of its `ignore_index`, which is -100 unless you set it otherwise: "Specifies a target value that is ignored and does not contribute to the input gradient." The mean is then taken over the other places. To read the tags afterwards, each word gets the tag with the highest score at its first piece.

    Jurafsky and Martin describe another way to train: "For training, we can just assign the gold-standard tag associated with each word to all of the subword tokens derived from it." To read the tags, they too take the first piece.
    """)
    return


@app.cell(hide_code=True)
def idea_heads():
    mo.Html(C.heads_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.vstack([
        mo.md(r"""
        ### Frozen or fine-tuned

        With BERT-Tiny frozen, its vectors are features, as in [A pretrained network as feature extractor](/course/open/torch-frozen-features?lang=en): only the head learns. Fine-tuned, every parameter learns. BERT's authors fine-tuned all of BERT, with learning rates of 0.00005, 0.00003 or 0.00002, batches of 16 or 32 examples and 2 to 4 epochs. For the 24 small models, BERT-Tiny among them, the notes in Google's BERT repository list larger learning rates, "3e-4, 1e-4, 5e-5, 3e-5", and 4 epochs. The demo in this lesson fine-tunes with a learning rate of 0.0001, and trains the head alone with 0.001.

        How much fine-tuning adds depends on the model, the task and what sits on top. Matthew Peters, Sebastian Ruder and Noah A. Smith compared the two ways for BERT-base, a model 25 times the size of BERT-Tiny. Frozen, their head was more than one layer: a weighted combination of all of BERT's layers went into a larger network of its own for the task. For the names in CoNLL 2003, a collection of news texts, they found a span F1 of 0.922 frozen and 0.924 fine-tuned; for sentences of movie reviews, an accuracy of 0.930 frozen and 0.935 fine-tuned. With only one layer on the frozen BERT-Tiny, this lesson measures far larger differences.

        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **head** (for a task) | a new layer on a pretrained model that turns its vectors into the scores of a task |
        | **token classification** | sequence labeling with a pretrained encoder: a head gives a label to the vector of every token |
        | **-100** | the label `nn.CrossEntropyLoss` leaves out: its `ignore_index` |
        | **[PAD]** | piece number 0: a short sentence is filled up with it to the length of the longest in the batch |
        | `model(ids, mask)` | BERT-Tiny on a batch: `ids` holds the numbers of the pieces, one row per sentence, and `mask` is True at the real places and False at the [PAD]s; it gives a list of 3 tensors of shape (sentences, places, 128), before the first block and after each block |
        | `x[:, k]` | place `k` of every row of the tensor `x`: from shape (n, t, 128) to (n, 128) |
        """),
        L.watch_out("-100 is not a tag. Tag number 0 is `O`, outside every name: a label of 0 would teach the model that the place is outside every name. A label of -100 teaches it nothing about that place."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Put a head on BERT-Tiny: `nn.Linear(128, 2)` on the vector at [CLS] to label a sentence, `nn.Linear(128, 7)` on every place for token classification.
    2. Give every training example its labels: one for a sentence; for token classification, a tag at the first piece of every word and -100 at every other place.
    3. Train with the cross-entropy: only the head, with BERT-Tiny frozen, or everything, with a small learning rate.
    4. Read the answers: the highest of the 2 scores for a sentence; for a word, the tag with the highest score at its first piece.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The button below downloads BERT-Tiny. The code then runs it on one sentence, *I love this device.* `bert.load()` gives BERT-Tiny itself, frozen, and `bert.tokenizer().encode` the numbers of the pieces of a text, with *[CLS]* and *[SEP]*. The sentence is a batch of one, so `mask` is True everywhere: no piece has the number 0, the number of [PAD]. The code puts a new head for 2 scores on the vector at [CLS] of the first sentence, and a new head for 7 scores on every place. Both heads have random numbers: they learned nothing yet.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download BERT-Tiny, 18 MB")
    return (get_button,)


@app.cell(hide_code=True)
def bert_cell(get_button):
    bert = C.bert_or_stop(get_button, {
        "ask": "*BERT-Tiny is not on your computer yet. The button downloads it once from Hugging Face, at a fixed version: `google/bert_uncased_L-2_H-128_A-2`, 18 MB, under the Apache-2.0 license. The worked example, the demo and the part where you try it yourself need it; the exercises do not.*",
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
    import torch
    from torch import nn

    model = bert.load()
    ids = torch.tensor([bert.tokenizer().encode("I love this device.")])
    mask = ids > 0
    print(bert.pieces("I love this device."))
    layers = model(ids, mask)
    print(len(layers), layers[-1].shape)
    torch.manual_seed(0)
    sentence_head = nn.Linear(128, 2)
    tag_head = nn.Linear(128, 7)
    with torch.no_grad():
        print(sentence_head(layers[-1][0, 0]))
        print(tag_head(layers[-1]).shape)
    print(bert.pieces("Ada Lovelace lived in London."))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The sentence has 7 pieces, *[CLS]* and *[SEP]* included, and `model` gives 3 tensors; the last has shape (1, 7, 128): 1 sentence, 7 places, 128 numbers. `layers[-1][0, 0]` is the vector at place 0 of sentence 0, [CLS], and the head for sentences gives it 2 scores. The head for tags gives 7 scores at each of the 7 places: shape (1, 7, 7). The pieces of *Ada Lovelace lived in London.* show what token classification has to deal with: *Lovelace* became *love* and *##lace*, and BERT-Tiny wrote every letter small.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_classify():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. classify: 2 scores for every sentence

        Write `classify(model, head, ids, mask)`. `ids` holds the numbers of the pieces of n sentences, one row per sentence, filled up with [PAD] to the same length; `mask` is True at the real places. Give back the 2 scores that `head` gives for the vector at [CLS] of every sentence, after the last block: a tensor of shape (n, 2).

        The check uses a small stand-in for BERT-Tiny, `model`, with 3 numbers per place instead of 128, and a head `head` from 3 numbers to 2 scores. It calls `classify(model, head, ids, mask)` with two sentences, the second with two [PAD]s, and with one sentence of five pieces. Then a case of its own: three sentences of different lengths, two of them with [PAD]s. Replace the `...`.
        """),
        L.task("Give back the 2 scores of the head for the vector at [CLS] of every sentence, after the last block."),
    ])
    return


@app.cell(hide_code=True)
def exercise_classify():
    box_classify = L.code_box("classify", lines=3)
    box_classify
    return (box_classify,)


@app.cell(hide_code=True)
def check_classify(box_classify):
    result_classify = L.check("classify", box_classify.value)
    result_classify
    return (result_classify,)


@app.cell(hide_code=True)
def hints_classify():
    L.hints(
        "classify",
        [
            "One line with three steps: run `model` on `ids` and `mask`, take the last of the 3 tensors it gives, and give `head` the vector at [CLS] of every sentence.",
            "The last tensor has shape (n, t, 3) for the stand-in, and (n, t, 128) for BERT-Tiny: sentence, place, number. [CLS] stands at place 0 of every sentence, and the words table shows how to take one place of every row at once.",
            "The one line: `    return head(model(ids, mask)[-1][___])`",
        ],
        why=r"`return head(model(ids, mask)[-1][:, 0])`: `model(ids, mask)[-1]` is the tensor after the last block, `[:, 0]` takes place 0, [CLS], of every sentence, and `head` turns each of those vectors into 2 scores. Without `mask`, the model would treat the [PAD]s as real pieces, and the vector at [CLS] of a short sentence would change.",
    )
    return


@app.cell(hide_code=True)
def text_after_classify():
    mo.md(r"""
    The picture uses your `classify` on BERT-Tiny's numbers for six held-out sentences, with a head the course trained on the frozen vectors at [CLS]: 5 of the 6 come out on the right side. *I did not expect this to be so good!*, which praises, gets 0.48.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your classify on BERT-Tiny: frozen, then fine-tuned

    The button below trains BERT-Tiny with a new head for 2 scores on the 2,400 training sentences, with your `classify` once it is right, else with the course's. It trains twice, each time for 4 epochs of 75 steps of 32 sentences: first with BERT-Tiny frozen, so that only the head learns, then with all of BERT-Tiny learning. That takes about a minute, the second run the most. A chart shows the loss of every step, and dashed the loss on the 600 held-out sentences after every epoch; in the second run, thin and dashed in gray, that of the first. Beside it, the probability of positive for the six sentences of the picture, and under it their vectors at [CLS] after the last block, both drawn again as BERT-Tiny trains. Then you can rewind through the second run.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    demo_button = mo.ui.run_button(label="Fine-tune BERT-Tiny")
    demo_button
    return (demo_button,)


@app.cell(hide_code=True)
def demo_tune(bert, box_classify, demo_button, result_classify):
    bert   # BERT-Tiny is downloaded: this cell waits for the download button
    demo_result = None
    if not demo_button.value:
        _out = mo.md("*Press the button to fine-tune BERT-Tiny.*")
    else:
        _now = {}   # the live chart of each run, made at its first report, and its held-out loss so far
        _titles = {"frozen": "Only the head learns", "tuned": "All of BERT-Tiny learns"}

        def _progress(_p):
            _run = _p["run"]
            if _run not in _now or _p["step"] == 1:   # a new run, also when the course's runs again in place of yours
                _now[_run] = L.live({"title": _titles[_run], "x": "step", "y": "loss",
                                     "alt": "The loss of every step so far, and dashed the loss on the 600 held-out sentences after every epoch; thin and dashed in gray, the held-out loss of the run with only the head learning." if _run == "tuned" else "The loss of every step so far, and dashed the loss on the 600 held-out sentences after every epoch."}, _p["steps"], low=0.0,
                                    key=[("loss", False, "2,400 training sentences, every step"), ("loss", True, "600 held-out sentences, after every epoch")])
                _now[_run].watch(lambda: C.now_parts(_p["look"](), _run, L.lang))
                _now[_run + "-held"] = [(0, _p["held_start"])]
                if _run == "tuned" and "frozen-held" in _now:   # the held-out loss with only the head learning, in gray
                    _now["tuned"].reference(_now["frozen-held"], "only the head", role="ink-2")
            _line = f"**{_titles[_run]}** · epoch {_p['epoch']} of {_p['epochs']} · step {_p['step']} of {_p['steps']} · loss {L.number(_p['loss'], 3)}"
            if "held_loss" in _p:   # the held-out sentences after every epoch, dashed
                _now[_run + "-held"].append((_p["step"], _p["held_loss"]))
                _line += f" · {_p['right']} of the 600 held-out sentences right"
            _now[_run].reference(_now[_run + "-held"], "", role="loss", replace=True)
            _now[_run].add(_p["step"], _p["loss"], _line)

        demo_result = C.demo(box_classify.value if result_classify.state == "correct" else None, _progress)
        _who = "your `classify`" if demo_result["yours"] else "the course's `classify`"
        _labels = {"title": "All of BERT-Tiny learns", "x": "step", "y": "loss", "frozen": "only the head",
                   "alt": "The loss of every step of the run in which all of BERT-Tiny learns, dashed its loss on the 600 held-out sentences after every epoch, and thin and dashed in gray that of the run with only the head learning."}

        def _frame(_s):
            _s = int(_s)
            _right = C.held_at(demo_result, _s)
            _said = f"After step {_s}: {_right} of the 600 held-out sentences right." if _right is not None else f"After step {_s}, in epoch {(_s - 1) // 75 + 1}."
            return C.rewind_frame(demo_result, _s, _labels, L.lang), _said

        _table = mo.md(f"""With {_who}, of the 600 held-out sentences, after 4 epochs:

| | right |
|---|---|
| only the head learns, BERT-Tiny frozen | {C.held_at(demo_result, C.STEPS, "frozen")} |
| all of BERT-Tiny learns: fine-tuned | {C.held_at(demo_result, C.STEPS)} |
| logistic regression on words, in *Logistic regression on words* | {C.REFERENCE['words']} |

Over 5 seeds, with the course's code: {C.REFERENCE['frozen'][0]} to {C.REFERENCE['frozen'][1]} with only the head learning, {C.REFERENCE['tuned'][0]} to {C.REFERENCE['tuned'][1]} fine-tuned.""")
        _out = mo.vstack([L.explore("rewind", "step", _frame, start=C.STEPS, values=C.rewind_steps(demo_result), role="",
                                    task="Go through the run from step 0: when do the six sentences move to their side, and do their rows of cells change with them?"), _table])
    _out
    return (demo_result,)


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    In the course's runs, the head alone labels 430 of the 600 held-out sentences right after 4 epochs, and fine-tuned BERT-Tiny 497. In the frozen run, the rows of cells stay the same at every step, and only the head moves the bars. In the fine-tuned run, the rows themselves change, and after epoch 2 five of the six sentences are on the right side. Fine-tuned BERT-Tiny gets about as many right as logistic regression on words, 491: a small model on 2,400 sentences gains little over a weight per word. In the last epoch, its held-out loss goes up again, from 0.402 to 0.491, while the number right barely moves, from 498 to 497.

    Both runs put *I did not expect this to be so good!* on the negative side. In *Logistic regression on words*, *not* had the strongest weight toward negative of all words, −1.78.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_labels():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. piece_labels: a label for every place

        Write `piece_labels(word_pieces, tags)`. `word_pieces` holds the pieces of every word of a sentence, a list per word, and `tags` the number of the tag of every word. Give back a label for every place BERT-Tiny reads: -100 for [CLS]; then for every word its tag at its first piece and -100 at every later piece; then -100 for [SEP].

        For `piece_labels([["ada"], ["love", "##lace"], ["lived"], ["in"], ["london"], ["."]], [1, 2, 0, 0, 3, 0])` the result is `[-100, 1, 2, -100, 0, 0, 3, 0, -100]`. Replace the `...`. The check calls that and the same for *Marie Curie moved to Paris.*, with `[["marie"], ["cu", "##rie"], ["moved"], ["to"], ["paris"], ["."]]`, then cases of its own: a word of three pieces, a sentence of one word, a sentence in which every word is one piece, with an organization, and a test sentence of Universal NER, `C.word_pieces(913)` with `C.gold_tags(913)`: *I just had the best experience at this Kal Tire location .*
        """),
        L.task("Give back -100 for [CLS], the tag of every word at its first piece and -100 at its other pieces, and -100 for [SEP]."),
    ])
    return


@app.cell(hide_code=True)
def exercise_labels():
    box_labels = L.code_box("piece_labels", lines=7)
    box_labels
    return (box_labels,)


@app.cell(hide_code=True)
def check_labels(box_labels):
    result_labels = L.check("piece_labels", box_labels.value)
    result_labels
    return (result_labels,)


@app.cell(hide_code=True)
def hints_labels():
    L.hints(
        "piece_labels",
        [
            "`labels` starts with -100, for [CLS]. Go through the words with `for i in range(len(word_pieces)):`. For word `i`, add one label for its first piece to `labels` with `labels.append`, then -100 for every piece in `word_pieces[i][1:]`. After the loop, one more -100, for [SEP].",
            "The first piece of word `i` carries the tag of word `i`. Which of the two arguments holds the tags, and which place in it belongs to word `i`?",
            "The line in the loop that labels the first piece: `        labels.append(___)`",
        ],
        why=r"`labels.append(tags[i])`: the first piece of word `i` gets its tag. In full, after `labels = [-100]`: `for i in range(len(word_pieces)):` with `labels.append(tags[i])` under it, and under that `for piece in word_pieces[i][1:]:` with `labels.append(-100)`; after the loop, `labels.append(-100)` for [SEP]. In the example, *love* gets 2, `I-PER`, and *##lace* gets -100.",
    )
    return


@app.cell(hide_code=True)
def text_after_labels():
    mo.md(r"""
    In the picture, your function labeled the places of three test sentences. Of their 41 places, 28 get a tag; the other 13, the two ends and the later pieces of *ginny*, *Travelocity*, *Expedia*, *carval* and *walmart*, get -100.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_first():
    mo.md(r"""
    ### Exercise 3. Why -100 at a later piece?
    """)
    return


@app.cell(hide_code=True)
def choice_first_cell():
    choice_first = L.choice(
        "first-piece",
        question="In training, *##lace*, the second piece of *Lovelace*, gets the label -100. Why?",
        options={
            "outside": "So that the model learns that *##lace* is outside every name",
            "novector": "Because BERT-Tiny gives no vector at a piece that continues a word",
            "right": "So that each word counts once, at the piece its tag is read from",
            "faster": "To make training faster, with fewer places to compute",
        },
        correct="right",
        why={
            "right": "With a tag at every piece, *Travelocity*, three pieces, would count three times in the loss, and *Paris* once. With -100, every word gives the loss one label, at its first piece, the place its tag is read from after training. Jurafsky and Martin describe the other way, the word's tag at every piece; to read the tags, both ways take the first piece.",
            "outside": "That would be the label 0, the tag `O`. What does `nn.CrossEntropyLoss` do with a place whose label is -100?",
            "novector": "BERT-Tiny gives a vector at every place, *##lace* too, and the head gives 7 scores there. What does -100 do with those scores in the loss?",
            "faster": "BERT-Tiny and the head compute at every place either way, and the loss is a small part of the work. What would change in the loss if *##lace* got the tag of *Lovelace*?",
        },
    )
    choice_first
    return (choice_first,)


@app.cell(hide_code=True)
def text_exercise_tags():
    mo.vstack([
        mo.md(r"""
        ### Exercise 4. word_tags: a tag for every word

        Write `word_tags(scores, word_pieces)`. `scores` is a NumPy array with a row of 7 scores for every place of a sentence, [CLS] and [SEP] included, as the head for tags gives them; `word_pieces` is as in exercise 2. Give back the number of the tag of every word: the tag with the highest score at its first piece.

        `ada_scores` holds scores of the course's own for *Ada Lovelace lived in London.*, 9 places: at each place one tag has 2.0 and the other tags 0.0. For `word_tags(ada_scores, [["ada"], ["love", "##lace"], ["lived"], ["in"], ["london"], ["."]])` the result is `[1, 2, 0, 0, 3, 0]`. Replace the `...`. The check calls that and the same for *Marie Curie moved to Paris.*, with `curie_scores`, then cases of its own: a word of three pieces, a sentence of one word whose first piece has two tags with the same highest score, where the lower number counts, and the scores of the course's fine-tuned BERT-Tiny for two test sentences, `C.test_scores(379)` with `C.word_pieces(379)` and `C.test_scores(913)` with `C.word_pieces(913)`.
        """),
        L.task("Give back, for every word, the number of the tag with the highest score at its first piece."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tags():
    box_tags = L.code_box("word_tags", lines=10)
    box_tags
    return (box_tags,)


@app.cell(hide_code=True)
def check_tags(box_tags):
    result_tags = L.check("word_tags", box_tags.value)
    result_tags
    return (result_tags,)


@app.cell(hide_code=True)
def hints_tags():
    L.hints(
        "word_tags",
        [
            "`tags` starts empty, and `place` at 1: the first piece of the first word, right after [CLS]. Go through the words with `for pieces in word_pieces:`. For each word, append `int(np.argmax(scores[place]))` to `tags`, then move `place` on to the first piece of the next word.",
            "A word of three pieces takes three places, and a word of one piece one place. How far does `place` move after a word?",
            "The line in the loop that moves on: `        place = place + ___`",
        ],
        why=r"`place = place + len(pieces)`: after a word, `place` jumps over all its pieces, to the first piece of the next word. In the example, *love* stands at place 2; *Lovelace* has two pieces, so the next word, *lived*, starts at place 4, and *##lace* at place 3 is skipped.",
    )
    return


@app.cell(hide_code=True)
def text_after_tags():
    mo.md(r"""
    In the picture, your function read the tags of all 1,471 test sentences of Universal NER from the scores of a BERT-Tiny that the course fine-tuned for names: with the head for tags, the labels of exercise 2, a learning rate of 0.0001 and 3 epochs over the 8,773 training sentences. It finds 783 names, 477 of them right, of the 802 names in the test set: a span F1 of 0.602. The tagger of [Finding names: named entity recognition](/course/open/nlp-named-entities?lang=en) got 0.551 in [Scoring a tagger: accuracy and span F1](/course/open/nlp-span-f1?lang=en); the same head on a frozen BERT-Tiny, trained with a learning rate of 0.01, gets 0.411. Mayhew and others, who made Universal NER, give 0.858 for a far larger model trained on the same English training set, on their test set, which also has the emails this course leaves out.

    BERT-Tiny writes every text in small letters before it cuts it into pieces: *Travelocity* and *travelocity* are the same to it. The tagger of *Finding names* looked at word shapes, with a capital as `X`. In the picture, fine-tuned BERT-Tiny misses *Travelocity* and *Expedia*, and calls *carval* and *walmart* places where the test set has organizations.
    """)
    return


@app.cell(hide_code=True)
def text_explore():
    mo.md(r"""
    ### Try it: a sentence of your own

    Type an English sentence, such as a line from a review. The picture shows the probability of positive that the two runs of the demo give it: the head on the frozen BERT-Tiny, and the fine-tuned BERT-Tiny. It needs the demo above. The sentence below praises with two negative words; try one with *but* in the middle, and the same sentence with its two halves swapped.
    """)
    return


@app.cell(hide_code=True)
def explore_controls():
    explore_text = mo.ui.text(value="Not bad at all.", label="Sentence", full_width=True)
    explore_text
    return (explore_text,)


@app.cell(hide_code=True)
def explore_look(bert, demo_result, explore_text):
    mo.stop(demo_result is None, mo.md("*Fine-tune BERT-Tiny with the button of the demo first.*"))
    mo.Html(C.explore_picture(bert, demo_result, explore_text.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** BERT-Tiny learned by filling in blanks in [BERT: filling in the blank](/course/open/nlp-bert?lang=en). A new head turns it into a model for a task: on the vector at [CLS] to label a sentence, on every place to tag every word, as in [A label per word: sequence labeling and BIO](/course/open/nlp-bio-tags?lang=en), with the tag at the first piece of a word and -100 at every other place. Fine-tuning trains the head and all of BERT-Tiny together, with a small learning rate; the two phases of [Reusing a trained network: freeze, then fine-tune](/course/open/torch-fine-tune?lang=en) first trained the head alone. On the reviews, fine-tuned BERT-Tiny labels 497 of the 600 held-out sentences right, against 430 with only the head learning; for names, it reaches a span F1 of 0.602, against 0.551 for the tagger with word shapes.

    **An everyday example.** RobBERT is a Dutch model of the BERT family. Pieter Delobelle and colleagues, who made it, fine-tuned it on reviews of books in Dutch, and it labels 95.1% of the reviews in their test set right as positive or negative. For the names in the Dutch texts of CoNLL-2002, it reaches a span F1 of 0.891.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_first, result_classify, result_labels, result_tags):
    L.wrap_up(result_classify, result_labels, choice_first, result_tags)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
