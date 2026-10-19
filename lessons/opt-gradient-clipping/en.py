# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson opt-gradient-clipping. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("opt-gradient-clipping", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en) and [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_half():
    quick_half = L.choice(
        "quick-half",
        question="You multiply every number of a vector by 0.5. What happens to its norm, and to its direction?",
        options={
            "right": "The norm halves, and the direction stays the same",
            "quarter": "The norm becomes a quarter, and the direction stays the same",
            "turns": "The norm halves, and the direction turns",
        },
        correct="right",
        why={
            "quarter": "The numbers are squared, and the squares do become a quarter, but the norm is the square root of their sum: the square root of a quarter is a half. See [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en).",
            "turns": "Its cosine similarity with the old vector stays 1: it looks only at the direction, and multiplying by a positive number does not change that. See [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en).",
            "right": "The arrow becomes half as long and points the same way: its cosine similarity with the old vector is 1. Today you use exactly that to make a gradient shorter without turning it.",
        },
    )
    quick_half
    return (quick_half,)


@app.cell(hide_code=True)
def quick_check_layers():
    quick_layers = L.choice(
        "quick-layers",
        question="On its way back through 20 layers, a gradient is multiplied by about 1.5 in each layer. It is 1 at the last layer. About how large is it at the first?",
        options={
            "sum": "About 30: 20 times 1.5",
            "same": "About 1.5: each layer changes it by the same factor",
            "right": "About 3,300: 1.5 multiplied by itself 20 times",
        },
        correct="right",
        why={
            "sum": "The factors are multiplied, not added: 1.5 times 1.5 times … , 20 times. See [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en).",
            "same": "The factor of each layer comes on top of those of the layers after it: they multiply. See [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en).",
            "right": "$1.5^{20} \\approx 3{,}300$: an exploding gradient. Weights with the right spread keep the factors near 1; today's lesson is the safety net for the steps where a gradient still becomes too large.",
        },
    )
    quick_layers
    return (quick_layers,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Even in a network that trains well, one mini-batch can give a gradient that is a hundred times larger than the others: an example with an error in it, or a steep spot in the loss. One step with such a gradient can throw the weights far from where they were. Large language models train for weeks on billions of examples, and somewhere among them is always a bad one. **Gradient clipping** guards every step: when the gradient is larger than a limit you choose, it is made shorter before the step.
    """)
    return


@app.cell(hide_code=True)
def predict_bad_batch():
    guess_bad_batch = L.guess(
        "bad-batch",
        question="The MLP of [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en) trains for one epoch of 94 mini-batches, with SGD at a learning rate of 0.5. In mini-batch 41, someone forgot to divide the pixels by 255: they run from 0 to 255. The norm of its gradient is 146; for the mini-batches before it, it is about 1. Without that mistake, the network reads 865 of the 1,000 test digits right. How many do you think it reads with it?",
        options={
            "same": "About 865: one mini-batch of 94 hardly counts",
            "right": "About 400: one huge step undoes much of what it learned",
            "zero": "About 100, as good as guessing: the loss becomes `nan` and stays there",
        },
        reveal="**405.** One step with a gradient 146 times too large throws the weights so far away that the loss jumps to over 1,000, and in the 53 mini-batches that are left the network does not get back to where it was. With the gradient clipped to a norm of 5, the same epoch, bad mini-batch included, reads 870 right.",
    )
    guess_bad_batch
    return (guess_bad_batch,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Put all numbers of the gradient, of all layers, one after the other in one long vector $\mathbf{g}$. Its norm $\lVert \mathbf{g} \rVert$ says how large the whole step would be. Choose a limit $c$. If the norm is above $c$, multiply every number of the gradient by $c / \lVert \mathbf{g} \rVert$; otherwise leave it alone:
        """),
        L.formula(r"\lVert [[grad:\mathbf{g}]] \rVert = \sqrt{\textstyle\sum_i [[grad:g_i]]^2}"),
        L.formula(r"[[grad:\mathbf{g}]] \leftarrow [[grad:\mathbf{g}]] \cdot \frac{[[hyper:c]]}{\lVert [[grad:\mathbf{g}]] \rVert} \quad \text{if } \lVert [[grad:\mathbf{g}]] \rVert > [[hyper:c]]"),
        mo.md(r"""
        After that, the norm is exactly $c$. Because every number is multiplied by the same factor, the gradient still points the same way: the step goes in the same direction, only less far. A normal step, below the limit, does not change at all.

        In PyTorch, one line between `loss.backward()` and `opt.step()` does it: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`. The `_` at the end of its name says that it changes the gradients themselves. It gives back the norm before clipping, as a tensor; logging it shows you when a bad mini-batch comes by, and how large the norms usually are, which helps to choose `max_norm`: a little above the usual norms, so that only the outliers are clipped. Large language models are usually trained with `max_norm=1.0`. There is also `torch.nn.utils.clip_grad_value_`, which cuts off every number on its own at $-c$ and $c$, a method you rarely need.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **gradient clipping** | | making the gradient shorter, before the step, when its norm is above a limit | `clip_grad_norm_` |
        | $\lVert \mathbf{g} \rVert$ | the norm of g | the norm of all numbers of the gradient together, of all layers | what `clip_grad_norm_` gives back |
        | $c$ | c | the limit: the largest norm the gradient may have | `max_norm` |
        """),
        L.watch_out("Clipping changes the step, not the loss: put it after `loss.backward()`, when the gradients are there, and before `opt.step()`, which uses them. And [Adam](/course/open/opt-adam?lang=en) needs it too. Adam keeps every step at about $\\eta$, so with Adam at 0.001 the bad mini-batch of the guess does less harm: 810 right instead of 845. But the huge gradient stays for a long time in Adam's averages $m$ and $s$, and the steps after it are worse: with clipping at 5, Adam reads 840."),
        mo.md(r"""
        The picture: the epoch of the guess, without clipping (dashed) and with clipping at 5 (solid). Left, the norm of the gradient of each mini-batch, before clipping: about 1, and 146 at mini-batch 41: with clipping, the only one above the limit. Without clipping, the next two mini-batches give norms of over 100 as well: the weights are far from where they were. Right, the loss of each mini-batch: without clipping, it jumps to over 1,000 and stays above 1, about 1.6 over the last ten mini-batches; with clipping, about 0.4. Both on a log scale.
        """),
        mo.Html(C.clip_picture({"norm_title": "Gradient norm", "loss_title": "Training loss", "x": "mini-batch", "norm": "norm before clipping", "loss": "loss",
                                "limit": "limit 5", "without": "without clipping", "with": "with clipping",
                                "alt": "Two panels on a log scale over 94 mini-batches. Left, the norm of the gradient: about 1, and 146 at mini-batch 41, far above the dashed limit at 5; without clipping it stays above 100 for two more mini-batches. Right, the loss: without clipping it jumps to about 1,000 and stays above 1; with clipping it has a short peak and goes on falling to about 0.4."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Gradient clipping by norm, in every training step:

    1. After `loss.backward()`, work out the norm of all gradients together: the square root of the sum of the squares of all their numbers.
    2. If it is above the limit $c$, work out the factor $c / \lVert \mathbf{g} \rVert$; otherwise the factor is 1.
    3. Multiply every gradient by that factor.
    4. Then take the step, `opt.step()`, and log the norm of step 1.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A network with a weight matrix and one bias term, whose gradient is clipped to $c = 5$, in NumPy. The comments with `#` say which step of the recipe each part is. Step 2 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_clip():
    import numpy as np

    _g_w = np.array([[3.0, 0.0], [0.0, 4.0]])
    _g_b = np.array([12.0])
    _c = 5.0
    # 1. the norm of all numbers together
    _norm = (np.sum(_g_w ** 2) + np.sum(_g_b ** 2)) ** 0.5
    # 2. complete: if _norm is above _c, make _scale _c / _norm
    _scale = 1.0
    # 3. every gradient times the factor
    _g_w = _g_w * _scale
    _g_b = _g_b * _scale
    print("norm:", _norm, "factor:", round(_scale, 4))
    print(np.round(_g_w, 4), np.round(_g_b, 4))
    print("norm after:", round((np.sum(_g_w ** 2) + np.sum(_g_b ** 2)) ** 0.5, 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The norm of the five numbers together is $\sqrt{9 + 16 + 144} = 13$, far above 5. As it is, the cell multiplies by 1, and the norm after is still 13. With step 2 complete, the factor is $5 / 13 \approx 0.3846$: the 12 becomes 4.6154, the 3 and the 4 become 1.1538 and 1.5385, and the norm after is 5. The numbers keep their ratio 3 : 4 : 12, so the gradient points the same way. On its own, the weight matrix has a norm of 5, at the limit: clipping each array by its own norm would leave it alone and shorten only the bias term, and the direction would change.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_direction():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The norm, not each number
    """)
    return


@app.cell(hide_code=True)
def choice_direction_cell():
    choice_direction = L.choice(
        "keeps-direction",
        question="`torch.nn.utils.clip_grad_value_` cuts off every number of the gradient on its own at $-c$ and $c$. Why do almost all training scripts clip the norm instead?",
        options={
            "less-work": "Clipping the norm is less work: it looks at one number instead of at all of them",
            "larger": "Cutting off each number can make the gradient larger than it was",
            "right": "Multiplying all numbers by one factor keeps the direction of the step; cutting off numbers one by one changes it",
        },
        correct="right",
        why={
            "less-work": "To work out the norm, you need every number too: the square root of the sum of all their squares.",
            "larger": "Cutting off only ever makes a number smaller in size, never larger. What it does change is the ratio between the numbers.",
            "right": "With $c = 5$, the gradient $(3, 4, 12)$ of the example becomes $(3, 4, 5)$: the large number is cut and the small ones are not, so the step turns. Clipping by norm gives $(1.15, 1.54, 4.62)$, the same direction.",
        },
    )
    choice_direction
    return (choice_direction,)


@app.cell(hide_code=True)
def text_exercise_clip():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Clipping a list of gradients

        Write `clip_grads(grads, max_norm)`: `grads` is a list of NumPy arrays, the gradients of the layers of a network, each with its own shape. Give back a list of the arrays clipped by recipe steps 1 to 3: one norm over all numbers of all arrays, and, if it is above `max_norm`, every array times `max_norm / norm`. Below the limit, give back the arrays as they are. The check calls `clip_grads([np.array([1, 2]), np.array([[2], [4]])], 2.5)`, which should give `[np.array([0.5, 1]), np.array([[1], [2]])]`, and tries other lists: below the limit, at the limit, one array, and all zeros.
        """),
        L.task("Make `clip_grads` clip a list of gradients by their norm together."),
    ])
    return


@app.cell(hide_code=True)
def exercise_clip():
    box_clip = L.code_box("clip-grads", lines=10)
    box_clip
    return (box_clip,)


@app.cell(hide_code=True)
def check_clip(box_clip):
    result_clip = L.check("clip-grads", box_clip.value)
    result_clip
    return (result_clip,)


@app.cell(hide_code=True)
def hints_clip():
    L.hints(
        "clip-grads",
        [
            "The norm of all arrays together: add up `np.sum(g ** 2)` for every `g` in `grads`, and take the square root of the total, with `** 0.5`.",
            "Then `if norm > max_norm:` make a new list and `append` `g * (max_norm / norm)` for every `g`, and give it back; otherwise give back `grads`.",
        ],
        why="`total = 0` and `for g in grads: total = total + np.sum(g ** 2)`; `norm = total ** 0.5`; `if norm > max_norm:` then `out = []`, `for g in grads: out.append(g * (max_norm / norm))` and `return out`; after the `if`, `return grads`. In the example, the norm is $\\sqrt{1 + 4 + 4 + 16} = 5$, so the factor is $2.5 / 5 = 0.5$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A training step with clipping

        Write `train_step(model, opt, xb, yb, max_norm)`: one training step on the mini-batch `xb`, `yb`, as in the loop of [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en), with gradient clipping to `max_norm` at the right place. Give back the norm before clipping, as a number: what `clip_grad_norm_` gives back, with `.item()`. The check makes a small `nn.Linear` model with an SGD optimizer, takes a step with a limit of 1, and in a second try two steps in a row, the first with a limit of 100 and the second with a limit of 2; it compares the norms you give back and the weights and bias terms after the steps.
        """),
        L.task("Make `train_step` take one training step with gradient clipping and give back the norm."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("train-step", lines=12)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("train-step", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "train-step",
        [
            "The steps of the project's loop stay: the loss, `opt.zero_grad()`, `loss.backward()` and `opt.step()`. Where between them are the gradients there, and not used yet?",
            "Between `loss.backward()` and `opt.step()` comes `torch.nn.utils.clip_grad_norm_`, with the parameters of the model and `max_norm`; what it gives back is the norm you need.",
        ],
        why="`loss = loss_fn(model(xb), yb)`; `opt.zero_grad()`; `loss.backward()`; `norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`; `opt.step()`; `return norm.item()`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The norm of [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en) measures how large a step would be, and scaling by one factor keeps its direction. [Vanishing and exploding gradients](/course/open/nn-vanishing-exploding?lang=en) showed how a gradient can grow layer by layer when the weights have the wrong spread; clipping catches a gradient that is too large, whatever its cause, such as one bad mini-batch. [Adam](/course/open/opt-adam?lang=en) limits each step itself, but its averages remember a huge gradient: clipping helps there too. In the next lesson, *Dropout*, you fight overfitting by switching off part of the neurons during training.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_direction, result_clip, result_step):
    L.wrap_up(choice_direction, result_clip, result_step)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
