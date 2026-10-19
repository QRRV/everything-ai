# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson lm-bigram-neural. Route: course/route/lessen/taal.yaml. Checks: checks.py.
# termen:allow regularization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("lm-bigram-neural", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [One-hot vectors for classes](/course/open/ml-one-hot?lang=en) and [Let the optimizer take the step](/course/open/torch-optimizer?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_eye():
    quick_eye = L.choice(
        "quick-eye",
        question="What does `np.eye(4)[2]` give?",
        options={
            "two": "`[0, 2, 0, 0]`: a 2 in place 2",
            "right": "`[0, 0, 1, 0]`: the one-hot vector of class 2",
            "row": "`[1, 1, 1, 1]`: row 2 of a matrix of ones",
        },
        correct="right",
        why={
            "two": "A one-hot vector holds a single 1, not the number of the class. See [One-hot vectors for classes](/course/open/ml-one-hot?lang=en).",
            "row": "`np.eye(4)` is the identity matrix: ones only on the diagonal. See [One-hot vectors for classes](/course/open/ml-one-hot?lang=en).",
            "right": "Row 2 of the identity matrix has its single 1 in place 2, counting from 0: the one-hot vector of class 2.",
        },
    )
    quick_eye
    return (quick_eye,)


@app.cell(hide_code=True)
def quick_check_zero():
    quick_zero = L.choice(
        "quick-zero",
        question="In a training loop, what does `opt.zero_grad()` do?",
        options={
            "weights": "It sets the weights back to 0, so that every step starts from the same table",
            "loss": "It sets the loss back to 0, so that it does not add up from step to step",
            "right": "It clears the slopes, so that `backward` does not add to the old ones",
        },
        correct="right",
        why={
            "weights": "The weights stay as they are; only their slopes are cleared. See [Let the optimizer take the step](/course/open/torch-optimizer?lang=en).",
            "loss": "The loss is worked out anew in every step anyway. `zero_grad` is about the slopes. See [Let the optimizer take the step](/course/open/torch-optimizer?lang=en).",
            "right": "`opt.zero_grad()` sets the `.grad` of every parameter to `None`. `backward` adds its slopes to what is already there, so without `zero_grad` every step would also carry the slopes of all the steps before.",
        },
    )
    quick_zero
    return (quick_zero,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your bigram model is a table you counted. Counting works for pairs, but a model that looks further back has far too many combinations to count, and a neural network does not count at all: it learns its numbers with gradient descent. In this lesson, a network with a single layer learns a bigram table from the same pairs, by gradient descent instead of counting. It is the first step from counting to learning, and every model in the rest of your project learns this way. This is the third step of your own LLM.
        """),
        draw.project_steps(["Count letters and pairs", "A bigram model by counting", "The same model, learned by a neural network",
                            "Your own tokenizer", "Your own GPT", "Pretrain a small LLM", "Teach it to follow instructions and preferences"],
                           2, here_label="← today", title="Where your project stands: your own LLM"),
        L.course_versions(),
        mo.md(r"""
        **An everyday example.** Two ways to know how long your trip to work takes: write down every trip and take the average, or start from a guess and change it a little after every trip, until it hardly changes any more.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_nll():
    guess_nll = L.guess(
        "net-nll",
        question=f"The network starts with a table of zeros: every character equally likely after every character, an NLL of 4.17. It trains for {C.STEPS} steps of gradient descent on the pairs of the first {C.grouped(C.START, L.lang)} characters of the plays. The table counted on those same characters gets an NLL of {C.number(C.counted_nll(), 2, L.lang)} there. What do you think the network gets?",
        options={
            "better": "About 1.5: learning beats counting",
            "right": "About 2.4: close to the counted table",
            "far": "About 3.5: still far from it",
        },
        reveal=f"**About 2.4: {C.number(C.net_nll(), 2, L.lang)}.** Close to the counted table, but not below it: on the pairs it learns from, no table predicts better than the counted one. Gradient descent does not find a better table; it finds the same table in another way, as the guess you change after every trip ends up at about the average.",
    )
    guess_nll
    return (guess_nll,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The network has one table of weights, $W$, 65 by 65: a row for every character, a column for every next character. To predict what comes after a character, it takes the one-hot row of that character, $X$, and multiplies it by the table: `X @ W`. A one-hot row has a single 1, so `X @ W` is one row of $W$: the row of that character. Its 65 numbers are logits; softmax turns them into the probabilities of the next character, and cross-entropy, the average NLL of the pairs, is the loss. So the network's table of probabilities is softmax of $W$, row by row.

        It learns $W$ with gradient descent: start with $W$ all zeros, so that every next character is equally likely, and take small steps downhill on the loss. Picking a row with a one-hot row gives the same as picking it by its number, so PyTorch can also write `W[x]`, for all pairs at once, without making the one-hot rows.

        Add a penalty for large weights to the loss, as in [Regularization and weight decay](/course/open/ml-regularization?lang=en), but with the mean of the squared weights instead of their sum: the mean is the sum divided by 4,225, the number of weights, so $\lambda$ need not be tiny. Large weights cost something, so the logits stay closer to 0, and every row of probabilities stays closer to even. That is what add-k smoothing of [Pairs you never saw: smoothing](/course/open/lm-smoothing?lang=en) did with the counts: a larger $\lambda$, like a larger k, gives flatter probabilities.
        """),
        L.formula(r"[[loss:\mathcal{L}]] = \text{cross-entropy}(X [[param:W]], y) + [[hyper:\lambda]] \cdot \text{mean}([[param:W]]^2)"),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | `F.cross_entropy(logits, y)` | | softmax and cross-entropy in one step, averaged over all pairs; `y` holds the number of the right next character of every pair. `F` is PyTorch's `torch.nn.functional`, imported with `import torch.nn.functional as F` | `F.cross_entropy(W[x], y)` |
        | `W[x]` | W of x | the rows of `W` for the characters in `x`, one row per pair: the same as `torch.eye(65)[x] @ W` | `W[x]` |
        | `torch.zeros(shape, requires_grad=True)` | | a tensor of zeros whose slopes PyTorch follows, as `np.zeros` but in PyTorch | `torch.zeros((65, 65), requires_grad=True)` |
        | `torch.softmax(W, dim=1)` | | softmax of every row of `W` | |
        """),
        L.watch_out("The loss does not get below the NLL of the counted table: on the pairs it learns from, no table predicts better than the counts. And even without the penalty, softmax never gives exactly 0: a pair that never came gets a very negative logit, and a very small probability."),
        mo.md(f"""
        The picture: the loss while the network trains on the first {C.grouped(C.START, L.lang)} characters of the plays, with the learning rate {C.LR}, and the NLL of the table counted on those same characters as a dashed line.
        """),
        mo.Html(C.loss_curve({"title": "Learning the table", "steps": "steps of gradient descent", "net": "the network",
                              "counted": f"the counted table, {C.number(C.counted_nll(), 2, L.lang)}", "fmt": lambda v: C.number(v, 1, L.lang),
                              "alt": f"A falling line, the loss of the network over {C.STEPS} steps: from 4.17 at the start it drops fast in the first steps, and then slowly towards the dashed line of the counted table at {C.number(C.counted_nll(), 2, L.lang)}, ending at {C.number(C.net_nll(), 2, L.lang)}."})),
        mo.md(f"""
        The loss drops fast in the first steps and then slowly. After {C.STEPS} steps it is {C.number(C.net_nll(), 2, L.lang)}, against {C.number(C.counted_nll(), 2, L.lang)} for the counted table. For frequent pairs, the two tables already agree: after "t", "h" has {C.number(C.net_prob('t', 'h'), 3, L.lang)} in the network and {C.number(C.counted_prob('t', 'h'), 3, L.lang)} in the counts. Rare characters learn more slowly: their rows get few pairs, and so small slopes.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To learn the bigram table with gradient descent:

    1. Turn the text into pairs of character numbers: `x` the one before, `y` the one after.
    2. Start with a table of zeros whose slopes PyTorch follows, `W = torch.zeros((65, 65), requires_grad=True)`, and an optimizer for it, `opt = torch.optim.SGD([W], lr=lr)`.
    3. Every step: the loss with its penalty, `loss = F.cross_entropy(W[x], y) + lam * (W ** 2).mean()`; then `opt.zero_grad()`, `loss.backward()` and `opt.step()`.
    4. The learned table of probabilities is `torch.softmax(W, dim=1)`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The pairs of "abab ba", with three characters: " " is 0, "a" is 1 and "b" is 2. 100 steps with the learning rate 5 and $\lambda$ = 0.01. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_learn():
    import torch
    import torch.nn.functional as F

    # 1. the pairs of "abab ba": x the character before, y the one after
    _x = torch.tensor([1, 2, 1, 2, 0, 2])
    _y = torch.tensor([2, 1, 2, 0, 2, 1])
    # 2. a table of zeros, and an optimizer for it
    _W = torch.zeros((3, 3), requires_grad=True)
    _opt = torch.optim.SGD([_W], lr=5)
    for _step in range(100):
        # 3. the loss with its penalty, then zero, backward and step
        _loss = F.cross_entropy(_W[_x], _y) + 0.01 * (_W ** 2).mean()
        _opt.zero_grad()
        _loss.backward()
        _opt.step()
    print(_loss.item())
    # 4. the learned table of probabilities
    with torch.no_grad():
        print(torch.softmax(_W, dim=1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The counted table of "abab ba" has the rows 0, 0, 1 for " " and for "a", and 1/3, 2/3, 0 for "b". The learned table comes close: about 0.96 and 0.98 where the counts have 1, and 0.33 and 0.66 for "b". Where the counts have 0, it has a little more than 0: 100 steps push those logits down only so far, and the penalty pulls them back towards 0. The first line is the loss of the last step, about 0.37: the cross-entropy, about 0.34, plus the penalty. The counted table has an NLL of about 0.32 on these pairs.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_row():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A one-hot row times the table
    """)
    return


@app.cell(hide_code=True)
def choice_row_cell():
    choice_row = L.choice(
        "one-hot-row",
        question="`W` is a 65 × 65 table, and `X` the one-hot row of character 3: a 1 in place 3 and 0 everywhere else. What is `X @ W`?",
        options={
            "col": "Column 3 of `W`: the weights into character 3",
            "right": "Row 3 of `W`: the logits after character 3",
            "num": "The single number `W[3, 3]`",
            "times": "All of `W`, times 3",
        },
        correct="right",
        why={
            "col": "A row on the left of `@` combines the rows of `W`: here only row 3, times 1. A column would come from `W @` a column.",
            "num": "`X` has 65 numbers, so `X @ W` has 65 too: a whole row, not one cell.",
            "times": "The 1 in place 3 picks a row; it does not multiply anything by 3.",
            "right": "Every row of `W` is multiplied by the number in `X` at its place, and only place 3 holds a 1: what is left is row 3.",
        },
    )
    choice_row
    return (choice_row,)


@app.cell(hide_code=True)
def text_exercise_loss():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The loss of a table

        `loss_of(W, x, y)` gives the loss of the table `W` on the pairs `x` and `y`, with one-hot rows as in *The idea*. The one-hot rows and the last line are there. Replace the `...` with the logits of every pair. The check calls `loss_of(torch.tensor([[0.0, 1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.5, 1.0]]), torch.tensor([0, 1, 2, 0]), torch.tensor([1, 0, 2, 2]))`, which should give about `0.7557`, and tries other tables. When your function is right, the check saves it in your project, as `work/projects/llm/bigram_net.py`.
        """),
        L.task("Make `loss_of` work out the logits of every pair from the one-hot rows `X` and the table `W`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_loss():
    box_loss = L.code_box("loss-of", lines=9)
    box_loss
    return (box_loss,)


@app.cell(hide_code=True)
def check_loss(box_loss):
    result_loss = L.check("loss-of", box_loss.value)
    result_loss
    return (result_loss,)


@app.cell(hide_code=True)
def hints_loss():
    L.hints(
        "loss-of",
        [
            "The first part of *The idea*: what does a one-hot row times the table give?",
            "`X` has one one-hot row per pair. Multiplying it with `W` gives one row of logits per pair.",
            "`logits = X @ ___`.",
        ],
        why="The rows of `W` for the characters 0, 1, 2 and 0 are the logits of the four pairs. Cross-entropy gives each the surprise of its right next character, 1, 0, 2 and 2, and takes the mean: about 0.7557.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_train():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Train the table

        `train(x, y, steps, lr, lam)` learns the table from the pairs `x` and `y`, given as lists of character numbers. The table of zeros, the optimizer and the loop are there. Replace the `...` with step 3 of the recipe. The check trains on the pairs of "the cat sat on the mat" for 20 steps with the learning rate 10 and `lam` 0, and compares your table with the course's by their NLL on those pairs, which should be about `0.6288`; it also tries other steps, learning rates and values of `lam`. When your function is right, the check saves it in your project, as `work/projects/llm/bigram_train.py`.
        """),
        L.task("Make `train` take one step of gradient descent on the loss with its penalty, every time round the loop."),
    ])
    return


@app.cell(hide_code=True)
def exercise_train():
    box_train = L.code_box("train", lines=14)
    box_train
    return (box_train,)


@app.cell(hide_code=True)
def check_train(box_train):
    result_train = L.check("train", box_train.value)
    result_train
    return (result_train,)


@app.cell(hide_code=True)
def hints_train():
    L.hints(
        "train",
        [
            "Step 3 of the recipe, as in the worked example: four lines inside the loop.",
            "First the loss, with the penalty `lam` times the mean of the squared weights. Then the three lines of [Let the optimizer take the step](/course/open/torch-optimizer?lang=en): clear the slopes, work them out, take the step.",
            "`loss = F.cross_entropy(W[x], y) + lam * (W ** 2).mean()`, then `opt.zero_grad()`, `loss.backward()` and `opt.___()`.",
        ],
        why="With `lam` 0, the loss is only the cross-entropy, and after 20 steps with the learning rate 10 the table already predicts the pairs of the short text well: an NLL of about 0.63, against 4.17 for the table of zeros.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### What your project can do now

    The cell below trains a table with your `train`, on the pairs of the first {C.grouped(C.START, L.lang)} characters of the plays, for {C.STEPS} steps with the learning rate {C.LR} and `lam` 0, and lets the learned table write with your `generate` of [Sampling: let the model write](/course/open/lm-sampling?lang=en), or the course's when yours is missing or broken. It works once exercises 2 and 3 are done, and training takes a few seconds.
    """)
    return


@app.cell(hide_code=True)
def demo_learn(result_loss, result_train):
    result_loss, result_train  # run again when exercise 2 or 3 saves your module
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Finish exercises 2 and 3 first: then your `loss_of` and your `train` are in your project, and this cell uses them.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*Training with your `train` takes more than 30 seconds. Does each step go over the pairs once, with `W[x]`?*",
            "train": "*The `train` in `work/projects/llm/bigram_train.py` no longer gives what the check expects, or stops with an error. Did you change that file? Delete it, and check exercise 3 again: its check then saves your function again.*",
            "net": "*The `loss_of` in `work/projects/llm/bigram_net.py` no longer gives what the check expects, or stops with an error. Did you change that file? Delete it, and check exercise 2 again: its check then saves your function again.*",
            "sampling": "*Writing with `generate` from `work/projects/llm/sampling.py` stops with an error. Did you change that file? Delete it, and check exercise 3 of [Sampling: let the model write](/course/open/lm-sampling?lang=en) again.*",
        }[_result])
    else:
        _who = "" if _result["yours"] else " The text comes from the course's `generate`: yours is missing or broken, as the note under *Where your project stands* says."
        _out = mo.vstack([
            mo.md(f"""Your learned table gets an NLL of {C.number(_result['nll'], 2, L.lang)} on the pairs it learned from, worked out with your `loss_of`, against {C.number(C.counted_nll(), 2, L.lang)} for the counted table. After "t", "h" has {C.number(_result['th'], 3, L.lang)}. What it writes, starting with "T", with seed 1:"""),
            mo.Html(C.text_block(_result["text"])),
            mo.md(f"""It writes like your counted model: it is the same kind of table, only learned instead of counted, and from a smaller part of the plays.{_who}"""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The one-hot rows of [One-hot vectors for classes](/course/open/ml-one-hot?lang=en), the cross-entropy of [Cross-entropy loss for many classes](/course/open/ml-cross-entropy?lang=en) and the optimizer of [Let the optimizer take the step](/course/open/torch-optimizer?lang=en) together learn the table you counted before, and the penalty of [Regularization and weight decay](/course/open/ml-regularization?lang=en) does what smoothing did. Your project `llm` now has `bigram_net.py` and `bigram_train.py` as well. In the next lesson, *Base model or chat model?*, you see what the language models you use every day do differently from the one you built.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_row, result_loss, result_train):
    L.wrap_up(choice_row, result_loss, result_train)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
