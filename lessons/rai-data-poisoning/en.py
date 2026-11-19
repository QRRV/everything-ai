# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-data-poisoning. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow data point, neural nets, convnet

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-data-poisoning", "en")
    C = L.checks   # the digits of QMNIST, the network of torch-mlp-digits, and labels the course flips


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about what this lesson builds on.
    """)
    return


@app.cell(hide_code=True)
def quick_check_stage():
    quick_stage = L.choice(
        "quick-stage",
        question="NIST sorts attacks on machine learning by the stage at which they strike. When does a poisoning attack strike?",
        options={
            "right": "While the model learns, through its training data",
            "use": "When the model is in use, through an input it reads",
            "file": "After training, by changing the file of weights",
        },
        correct="right",
        why={
            "right": "NIST tells the two families apart by the stage: poisoning strikes while the model learns, evasion while it is used. See *When: while it learns, or while it is used* in [Attacks on machine learning](/course/open/rai-ml-threat-models?lang=en).",
            "use": "That is an evasion attack, such as the stickers on the stop sign: the model stays as it was trained. See *When: while it learns, or while it is used* in [Attacks on machine learning](/course/open/rai-ml-threat-models?lang=en).",
            "file": "A poisoning attack does not need the weights: it works through the data the model learns from. See *When: while it learns, or while it is used* in [Attacks on machine learning](/course/open/rai-ml-threat-models?lang=en).",
        },
    )
    quick_stage
    return


@app.cell(hide_code=True)
def quick_check_argmax():
    quick_argmax = L.choice(
        "quick-argmax",
        question="DigitsMLP gives 10 logits for each image. Which digit does it read?",
        options={
            "right": "The digit whose logit is the largest",
            "sum": "The sum of the ten logits, rounded to a whole digit",
            "half": "The first digit whose logit is above 0.5",
        },
        correct="right",
        why={
            "right": "`model(X_test).argmax(dim=1)` takes, for each image, the place of the largest logit, and that place is the digit. See the worked example of [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en).",
            "sum": "The ten logits are scores, one for each digit; their sum says nothing about which digit it is. The digit is the place of the largest: `argmax`. See [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en).",
            "half": "Logits are not probabilities, and several of them can be above 0.5. The digit is the place of the largest: `argmax`. See [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en).",
        },
    )
    quick_argmax
    return


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model learns what its labels say. The made-up library wants a network that reads the handwritten numbers on its return slips, and asks volunteers to label 6,000 handwritten digits for it. One of them may want to cause trouble. That volunteer cannot touch the network or its code; they can only write wrong labels. How many wrong labels does it take, and does it matter which ones? In this lesson you flip labels of the digits yourself, train the network of [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en) on them, and measure what it reads, digit by digit. Nothing leaves your computer: you change your own copy of the labels.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="Trained on the clean labels, the network reads 929 of the 1,000 test digits right. Now 30% of the training labels are changed at random, each to another digit. How many does the same training read right?",
        options={
            "fewer": "About 650: 30% fewer",
            "right": "Still more than 850",
            "chance": "About 100, as many as guessing",
        },
        reveal="**881.** Flipping 1,792 of the 6,000 labels at random costs 48 test digits. The worked example shows it, and *The idea* says why a network copes with so much noise.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### What poisoning is

    NIST's report on adversarial machine learning puts it broadly: "Poisoning attacks are broadly defined as adversarial attacks during the training stage of the ML algorithm." In data poisoning, "an adversary controls a subset of the training data by either inserting or modifying training samples." It is not new: "The first known poisoning attack was developed for worm signature generation in 2006".

    ### Worse everywhere, or wrong just there

    Poisoning can serve two of the goals of [Attacks on machine learning](/course/open/rai-ml-threat-models?lang=en): "Availability poisoning attacks typically cause indiscriminate degradation of the ML model on all samples, while targeted and backdoor poisoning attacks induce integrity violations on a small set of target samples." The first kind is easy to see: "Availability poisoning attacks are usually detectable by monitoring the standard performance metrics of ML models". The second shows only on the inputs the attacker chose.

    ### Label flipping

    The simplest way in is through the labels. NIST: "A simple black-box poisoning attack strategy is LABEL FLIPPING, in which an adversary generates training examples with incorrect or altered labels". To spoil a model as a whole, "This method may require a large percentage of poisoning samples". Against a target it is easy: "If the adversary can control the labeling function of the training data, then label-flipping is an effective targeted poisoning attack: the adversary simply inserts several poisoned samples with the target label, and the model will learn the wrong label."

    ### Random flips, chosen flips

    Which labels are flipped matters more than how many. Xiao, Xiao and Eckert (2012) flipped labels for a support vector machine, another kind of classifier, which [The widest street: the maximum-margin classifier](/course/open/ml-max-margin?lang=en) explains. At random, "the error rate hardly changes with 20 flipped labels". With 20 flips that their method chose, "(i.e. 10% of training data)", the error rose to 50%, "which is turned into the random guess." Biggio, Nelson and Laskov (2012) went further. For a support vector machine that tells two MNIST digits apart, such as 7 and 1, from 100 training images, they computed one training point: "a single attack data point caused the classification error to rise from the initial error rates of 2–5% to 15–20%."

    Neural networks cope with a lot of random noise. Rolnick and colleagues (2017) added extra examples with random labels to MNIST. For a convolutional network, a kind of network for images, they report: "Even with 100 noisy labels for every clean label the ConvNet still attains a performance of 91%." They put it down to a margin: "deep neural nets are robust even to structured noise, as long as the correct label remains the most likely by at least a small margin." Their noise came on top of every clean example. Flipping a share takes clean examples away, and they warn "that a decrease in the number of correct examples is more destructive to learning than an increase in the number of noisy labels."

    The margin is arithmetic. Flip a share $p$ of the labels at random, each to one of the nine other digits: of the images of each digit, $1 - p$ keep their own label, and $p / 9$ get any one other label. The right label stays the most common while $1 - p > p / 9$, that is while $p < 0.9$. The picture below draws both lines.

    ### Backdoors are something else

    A backdoor works differently: the attacker adds a trigger to some training inputs and gives them the label the attacker wants. Gu, Dolan-Gavitt and Garg (2017) trained BadNets with triggers such as "a single bright pixel in the bottom right corner of the image": on MNIST a BadNet "successfully mislabels > 99% of backdoored images", and "the attack succeeds even if backdoored images represent only 10% of the training dataset." Without the trigger it reads almost as well as before, so clean test digits do not show it. This lesson flips labels only.

    ### At the scale of the web

    Large models learn from more data than anyone checks by hand. Carlini and colleagues (2024) showed that, by buying old web domains for 60 US dollars, they "could have poisoned 0.01% of the LAION-400M or COYO-700M datasets in 2023", two large collections of images from the web. They did not: "We do not actually poison any datasets." Souly and colleagues (2025) poisoned language models during their training with documents that teach them to write gibberish after a trigger: "250 poisoned documents similarly compromise models across all model and dataset sizes". For their model of 13 billion parameters, that was 0.00016% of the training tokens; for one of 600 million, 0.0035%. The gibberish is harmless, and the blog post with the paper adds: "It remains unclear how far this trend will hold as we keep scaling up models."

    ### How sure is a score?

    A score on test digits is a measurement too. Yadav and Bottou (2019), who rebuilt MNIST as QMNIST, the digits of this course, write: "an error rate close to 1.0% measured on the usual 10,000 test example is reported as a 1% ± 0.2% error rate, that is, 100 ± 20 misclassifications." That margin of error is two standard errors. The standard error of a share $s$ measured on $n$ test images is $\sqrt{s(1 - s)/n}$: how much the share would vary if you measured it again on other test images. [How sure is a number? Confidence intervals and the bootstrap](/course/open/stats-bootstrap?lang=en) goes deeper. The margin of error is twice that, $2\sqrt{s(1 - s)/n}$. On the course's 1,000 test digits, a score of 0.929 has a margin of error of 0.016; on the 99 test images of the 7s, the margin of error of their score is 0.048. A drop smaller than the margin of error may be chance.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | | |
        |---|---|
        | **data poisoning** | an attack through the training data: the attacker adds training examples or changes them |
        | **label flipping** | data poisoning that changes only labels: at random, or of one digit to another |
        | **targeted poisoning** | poisoning that makes the model wrong on a few inputs the attacker chose, and leaves the rest as it was |
        | **backdoor** | a hidden behavior that a trigger in the input switches on |
        | **margin of error** | two standard errors, as QMNIST's paper uses it: how far a score on $n$ test images can be off by chance |
        | $s$, $n$ | a share measured on test images, and how many test images there are |
        """),
        L.watch_out("A score overall does not say where a model fails. The network of exercise 1, which learned the 7s as 1s, still reads 840 of the 1,000 test digits right, 89 fewer than clean: it loses 90 of the 7s, and reads the other digits as well as the clean network."),
        mo.Html(C.idea_picture("en")),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Keep the clean labels: flip a copy, so that you can train on both.
    2. Flip: a share of all labels at random, or the first share of one digit to another.
    3. Train the same network with the same seeds on the flipped labels.
    4. Measure it on the clean test digits: overall, and per digit.
    5. Compare each score with the clean one: a drop larger than the margin of error is more than chance.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Steps 1 to 4 for flips at random. `flip_random(labels, share, seed)` gives a copy in which each label, with chance `share`, moves 1 to 9 places on, past 9 back to 0, so that it becomes another digit. `% 10` is the remainder after dividing by 10, so 9 + 3 becomes 2. `.astype(np.int64)` makes the labels whole numbers of 64 bits, the type the loss wants, and `.numpy()` turns what the network reads back into a NumPy array. The network and its training are those of [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en): `DigitsMLP(64)`, 10 epochs of mini-batches of 64, with $\eta$ = 0.5, the same seeds every time. The cell trains it twice, on the clean labels and with 30% of them flipped, and counts the test digits each network reads right. It takes about two seconds.
    """)
    return


@app.cell
def example_flip():
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    # 1. the digits, and the clean labels
    data = np.load("data/qmnist-small.npz")
    X = torch.tensor(data["train_images"], dtype=torch.float32) / 255
    labels = data["train_labels"].astype(np.int64)
    X_test = torch.tensor(data["test_images"], dtype=torch.float32) / 255
    test_labels = data["test_labels"].astype(np.int64)


    # 2. a copy in which each label, with chance share, becomes one of the nine other digits
    def flip_random(labels, share, seed):
        rng = np.random.default_rng(seed)
        flipped = labels.copy()
        picked = rng.random(len(labels)) < share
        shifts = rng.integers(1, 10, size=len(labels))   # 1 to 9 places on: never the same digit
        flipped[picked] = (labels[picked] + shifts[picked]) % 10
        return flipped


    class DigitsMLP(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.hidden = nn.Linear(784, hidden)
            self.out = nn.Linear(hidden, 10)

        def forward(self, x):
            return self.out(torch.relu(self.hidden(x)))


    # 3. the same network and the same seeds, on labels y; it gives what it reads for each test digit
    def train(y):
        torch.manual_seed(0)
        model = DigitsMLP(64)
        opt = torch.optim.SGD(model.parameters(), lr=0.5)
        loss_fn = nn.CrossEntropyLoss()
        torch.manual_seed(2)
        loader = DataLoader(TensorDataset(X, torch.tensor(y)), batch_size=64, shuffle=True)
        for epoch in range(10):
            for xb, yb in loader:
                loss = loss_fn(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            return model(X_test).argmax(dim=1).numpy()


    # 4. the clean labels, and 30% of them flipped at random
    flipped = flip_random(labels, 0.3, 1)
    print("labels changed:", np.sum(flipped != labels), "of", len(labels))
    for name, y in [("clean", labels), ("30% flipped", flipped)]:
        pred = train(y)
        print(name, "right:", np.sum(pred == test_labels), "of", len(test_labels))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With 30% at random, 1,792 of the 6,000 labels change. The clean network reads 929 of the 1,000 test digits right, and the one trained on the flipped labels 881: 48 fewer, while almost one label in three was wrong. The course trained the same network on other shares too, with the same `flip_random` and seed 1:

    | share flipped at random | 0 | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 0.9 |
    |---|---|---|---|---|---|---|---|
    | labels changed | 0 | 602 | 1,203 | 1,792 | 2,982 | 4,181 | 5,386 |
    | test digits right | 929 | 900 | 887 | 881 | 788 | 614 | 102 |

    Up to half the labels, the network keeps most of what it reads. At 0.9, where the right label is no more common than each wrong one, it reads 102 right, about as many as guessing. Exercise 1 flips the labels of one digit instead, and exercise 2 measures what that does to each digit.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_flip():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. flip_digit: only the 7s

        An attacker who wants the network to misread one digit flips only that digit. `flip_digit(labels, source, target, share)` gives a copy of `labels` in which the first `share` of the images labeled `source` are labeled `target`, a count you round with `round`; every other label stays as it was. `flipped = labels.copy()` makes the copy. `np.arange(len(labels))` numbers the places 0, 1, 2 and so on; a mask keeps the places where the label is `source`. Replace the `...`. The check calls `flip_digit(LABELS, 7, 1, 0.8)`, then other digits and shares.
        """),
        L.task("Give the first share of the images labeled source the label target, in a copy of labels."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flip():
    box_flip = L.code_box("flip-digit", lines=10)
    box_flip
    return (box_flip,)


@app.cell(hide_code=True)
def check_flip(box_flip):
    result_flip = L.check("flip-digit", box_flip.value)
    result_flip
    return (result_flip,)


@app.cell(hide_code=True)
def hints_flip():
    L.hints(
        "flip-digit",
        [
            "Step 2 of the recipe, for one digit: find the places where the label is source, take the first part of them, and give those the label target.",
            "With a mask on the numbered places you get the places of source. How many to flip is `share` times how many places there are, rounded with `round`; a slice from the start takes that many.",
            "`places = np.arange(len(labels))[labels == source]`, `count = round(share * len(places))`, then `flipped[___] = target`",
        ],
        why="`np.arange(len(labels))[labels == source]` keeps the places of the images labeled `source`, in order. `round(share * len(places))` is how many of them to flip, and `flipped[places[:count]] = target` gives the first `count` of them the label `target` in the copy; `labels` stays as it was. With `flip_digit(LABELS, 7, 1, 0.8)`, 521 of the 651 sevens are labeled 1.",
    )
    return


@app.cell(hide_code=True)
def text_after_flip():
    mo.md(r"""
    That is 521 labels, fewer than the 602 that 10% at random changes. The course trained the network on them: it reads 840 of the 1,000 test digits right. Of the 99 test 7s, it reads 3 right and 89 as a 1. The 901 other test digits it reads as well as the clean network does: 837 of them, against 836. So the 521 labels cost 89 test digits, more than the 1,792 flipped at random cost (48), as the 7s are a tenth of the test digits. The overall score shows that something is wrong, but not what.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_per():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. per_digit: a score for each digit

        One score hides where a network fails. `per_digit(pred, labels)` gets what a network reads for each test image, `pred`, and the right digits, `labels`, and gives a list of ten shares: for each digit from 0 to 9, the share of the images labeled with that digit that `pred` reads right. Replace the `...`. The check calls `per_digit(GUESSES, TRUTH)`, twenty made-up guesses for twenty images, two of each digit, and then other guesses.
        """),
        L.task("Give for each digit the share of its images that pred reads right."),
    ])
    return


@app.cell(hide_code=True)
def exercise_per():
    box_per = L.code_box("per-digit", lines=10)
    box_per
    return (box_per,)


@app.cell(hide_code=True)
def check_per(box_per):
    result_per = L.check("per-digit", box_per.value)
    result_per
    return (result_per,)


@app.cell(hide_code=True)
def hints_per():
    L.hints(
        "per-digit",
        [
            "Step 4 of the recipe, per digit: for each digit, look only at the images whose label is that digit.",
            "A mask with `labels == digit` keeps those images. Of their guesses in `pred`, the share that is that digit too is the mean of `True` and `False`, which `np.mean` gives.",
            "`mine = labels == digit` in the loop over `range(10)`, then `shares.append(float(np.mean(___ == digit)))`",
        ],
        why="`for digit in range(10):` goes over the digits; `mine = labels == digit` is `True` for the images with that label, `pred[mine]` are their guesses, and `np.mean(pred[mine] == digit)` the share of them that is right. `float` makes a plain number of it. `per_digit(GUESSES, TRUTH)` gives 0.5 for the 7s and the 9s and 1.0 for every other digit.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_verdict():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. verdict: a drop, or chance?

        `verdict(clean, poisoned, n)` gets two scores on the same `n` test images: that of the clean network and that of a poisoned one. It gives `"worse"` when `poisoned` is lower than `clean` by more than the margin of error of `clean`, and `"chance"` otherwise. The margin of error is $2\sqrt{\text{clean}\,(1 - \text{clean})/n}$, with `math.sqrt`. It is a rough rule: it does not count that the poisoned score is a measurement too. Replace the two `...`. The check calls `verdict(0.929, 0.881, 1000)`, the clean network against the one with 30% flipped at random, and then other scores.
        """),
        L.task("Give \"worse\" when poisoned is below clean by more than the margin of error, else \"chance\"."),
    ])
    return


@app.cell(hide_code=True)
def exercise_verdict():
    box_verdict = L.code_box("verdict", lines=9)
    box_verdict
    return (box_verdict,)


@app.cell(hide_code=True)
def check_verdict(box_verdict):
    result_verdict = L.check("verdict", box_verdict.value)
    result_verdict
    return (result_verdict,)


@app.cell(hide_code=True)
def hints_verdict():
    L.hints(
        "verdict",
        [
            "Step 5 of the recipe: work out the margin of error of the clean score, then see whether the drop is larger.",
            "The margin of error is 2 times the square root of clean times 1 minus clean, divided by n. Only a drop counts, not a rise.",
            "`margin = 2 * math.sqrt(clean * (1 - clean) / n)`, then `if ___ > margin:` with `return \"worse\"` under it",
        ],
        why="`2 * math.sqrt(clean * (1 - clean) / n)` is two standard errors of the clean score. `clean - poisoned` is the drop; when it is larger than the margin of error, `verdict` gives `\"worse\"`, else the last line gives `\"chance\"`. For 0.929 on 1,000 test images the margin of error is 0.016, so 0.881 is worse.",
    )
    return


@app.cell(hide_code=True)
def text_after_verdict():
    mo.md(r"""
    Digit by digit, the picture shows the two attacks apart. With 80% of the 7s labeled 1, only the 7s are worse. With 30% at random, six digits are, none of them by nearly as much as the 7s: the worst, the 3s, are read 0.757 right, against 0.925 for the clean network.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Poison it yourself

    Choose how to flip the labels, at random or only the 7s as 1s, and which share. The 7s go through your `flip_digit` once exercise 1 is right, and through the course's until then. The button trains the network on those labels as the worked example does, in a few seconds. While it trains, the chart shows every 10 mini-batches the share of the test digits it reads wrong, with the clean network dashed; beside it you see the share of each digit it reads right, and what it reads for one test digit of each kind. Afterwards you can go back through the epochs.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_kind = mo.ui.dropdown(options={"at random": "random", "only the 7s, as 1s": "sevens"}, value="only the 7s, as 1s",
                               label="Flip")
    demo_share = L.slider("demo-share", "share flipped", start=0.8, low=0.0, high=1.0, step=0.1, role="hyper")
    demo_button = mo.ui.run_button(label="Train on these labels")
    mo.vstack([mo.hstack([demo_kind, demo_share], justify="start", gap=1.5), demo_button])
    return demo_button, demo_kind, demo_share


@app.cell(hide_code=True)
def demo_run(box_flip, demo_button, demo_kind, demo_share, result_flip):
    if not demo_button.value:
        _out = mo.md("*Choose the labels, and press the button to train.*")
    else:
        _kind, _share = demo_kind.value, round(float(demo_share.widget.value), 1)
        _yours = C.mine(box_flip.value, result_flip.state == "correct", "flip_digit") if _kind == "sevens" else None
        _what = f"{round(100 * _share)}% of the labels flipped at random" if _kind == "random" else f"{round(100 * _share)}% of the 7s labeled 1"
        _words = {"title": f"Trained with {_what}", "x": "epoch", "y": "share of test digits wrong", "digit": "digit",
                  "clean": "clean labels", "now": "now", "guess": "What the network reads now",
                  "run": "these labels", "after": "after epoch {e}", "guess_after": "What it read after epoch {e}",
                  "alt": f"The share of the 1,000 test digits that the network reads wrong, every 10 mini-batches, trained with {_what}; dashed, the network trained on the clean labels."}
        _now = {}

        def _progress(_p):
            if "live" not in _now:
                _now["live"] = L.live({"title": _words["title"], "x": "epoch", "y": _words["y"], "alt": _words["alt"]},
                                      C.EPOCHS, low=0.0, high=1.0)
                _now["live"].reference(C.course_run("random", 0.0)[0], "clean labels", role="loss")
            _now["live"].watch(lambda: C.now_parts(_p["pred"], _words, L.lang))
            _now["live"].add(_p["x"], _p["wrong"], f"**{_words['title']}** · epoch {_p['epoch']} of {C.EPOCHS} · {round(1000 * (1 - _p['wrong']))} of 1,000 test digits right")

        mo.output.replace(mo.md("*Epoch 1 · starting …*"))
        _run = C.demo(_kind, _share, _yours, _progress)
        if isinstance(_run, str):
            _note = {"flip": "*Your `flip_digit` stops with an error on the training labels, or gives no 6,000 labels from 0 to 9. The picture under exercise 1 says what goes wrong.*",
                     "slow-flip": "*Your `flip_digit` takes more than ten seconds on the 6,000 labels. Does it go over the labels only once?*",
                     "slow": "*The training takes more than two minutes. Is the computer busy with something else?*"}[_run]
            _out = _now["live"].stopped(_note) if "live" in _now else mo.md(_note)
        else:
            _labels, _curve, _epochs = _run
            _pred = _epochs[-1]
            _right = round(1000 * C.accuracy(_pred))
            _changed = int((_labels != C.LABELS.value).sum())
            _base, _shares, _n = C.per_digit(C.clean(), C.TEST_LABELS.value), C.per_digit(_pred, C.TEST_LABELS.value), C.counts()
            _worse = [str(_d) for _d in range(10) if C.verdict(_base[_d], _shares[_d], _n[_d]) == "worse"]
            _which = "the " + ", ".join(_d + "s" for _d in _worse[:-1]) + (" and " if len(_worse) > 1 else "") + (_worse[-1] + "s" if _worse else "")
            _drop = (f"By the course's `verdict`, the score drops by more than its margin of error for {_which}." if _worse
                     else "By the course's `verdict`, no digit drops by more than its margin of error.")
            _who = " Your `flip_digit` made the labels." if _yours else ""
            _out = mo.vstack([
                mo.md(f"""Trained with {_what}: {_changed:,} of the 6,000 labels changed. The network reads <span class="ea-num">{_right}</span> of the 1,000 test digits right, against {round(1000 * C.accuracy(C.clean()))} for the clean labels. {_drop}{_who}"""),
                L.explore("rewind", "epoch", lambda _e: (C.rewind_frame(_curve, _epochs, int(_e), _words, L.lang),
                                                         f"After epoch {int(_e)}: {round(1000 * C.accuracy(_epochs[int(_e) - 1]))} of the 1,000 test digits right."),
                          start=C.EPOCHS, values=list(range(1, C.EPOCHS + 1)), role="",
                          task=("Go back to epoch 1, then forward again: when does the network start to read the 7s as 1s?" if _kind == "sevens"
                                else "Go back to epoch 1, then forward again: which digits does it read right first?")),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_noise():
    mo.md(r"""
    ### Exercise 4. Why so little?

    Rolnick and colleagues kept 91% on MNIST with 100 random labels for every clean one. The network of this lesson, with 90% of its labels flipped at random, reads 102 of the 1,000 test digits right.
    """)
    return


@app.cell(hide_code=True)
def choice_noise_cell():
    choice_noise = L.choice(
        "noise",
        question="What explains the difference?",
        options={
            "right": "At 0.9, the right label is no more common than each wrong one",
            "big": "Their network was larger, and so it could simply ignore the wrong labels",
            "easy": "Their MNIST digits were easier to read than QMNIST",
        },
        correct="right",
        why={
            "right": "With 100 random labels for each clean one, every clean example stays, and the right label still comes out ahead: \"only 11 out of 101 labels are correct, as compared with 10.1 for pure chance\". Flipping 90% at random takes nine in ten clean examples away and leaves the right label as common as each of the nine others: the margin is gone.",
            "big": "What decides it is the margin: networks learn the right label \"as long as the correct label remains the most likely by at least a small margin.\" At 0.9 it no longer is. See *Random flips, chosen flips* in *The idea*.",
            "easy": "QMNIST rebuilds MNIST: \"The reconstructed training set contains 60,000 images matching each of the MNIST training images.\" The difference is in the labels. See *Random flips, chosen flips* in *The idea*.",
        },
    )
    choice_noise
    return (choice_noise,)


@app.cell(hide_code=True)
def text_exercise_count():
    mo.md(r"""
    ### Exercise 5. A share, or a number?

    Souly and colleagues poisoned language models of 600 million and of 13 billion parameters, each with 250 documents.
    """)
    return


@app.cell(hide_code=True)
def choice_count_cell():
    choice_count = L.choice(
        "count",
        question="What does that suggest for an attacker on the web?",
        options={
            "right": "A fixed number of documents can be enough, however big the data",
            "share": "The share of the data counts, so a big model needs far more",
            "lost": "Large models learn from so much text that 250 documents get lost in it",
        },
        correct="right",
        why={
            "right": "250 documents were 0.00016% of the training tokens of the model of 13 billion parameters and 0.0035% of that of 600 million, and both learned the backdoor. It was harmless gibberish, and whether this holds for still larger models is open, as the blog post with the paper says.",
            "share": "Older work measured poison as a share of the training data; Souly and colleagues found that the count matters: the same 250 documents worked at every size they tried. See *At the scale of the web* in *The idea*.",
            "lost": "That is what you might expect, and not what Souly and colleagues found: 250 documents compromised models of every size, \"despite the largest models training on more than 20 times more clean data.\" See *At the scale of the web* in *The idea*.",
        },
    )
    choice_count
    return (choice_count,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Attacks on machine learning](/course/open/rai-ml-threat-models?lang=en), an attacker changed an input after training; here the attacker changed the training data before it. Both follow NIST's goals: flips at random go for availability, flips of one digit for integrity. The network is that of [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en), with the same seeds, so every difference you measured comes from the labels. The next lessons of this module attack language models, starting with *How often does an injection work?*

    **An everyday example.** A shop with reviews. A hundred fake one-star reviews spread over all products hardly move the average of any of them; the same hundred on one product sink that product. The shop's average over all reviews falls by the same amount either way: only the average per product shows which one was hit.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_count, choice_noise, result_flip, result_per, result_verdict):
    L.wrap_up(result_flip, result_per, result_verdict, choice_noise, choice_count)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
