# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-formula-to-code. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-formula-to-code", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Reading a paper in three passes](/course/open/res-three-pass?lang=en) and [Gradients of matrix expressions: follow the shapes](/course/open/math-matrix-calculus?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_pass():
    quick_pass = L.choice(
        "quick-pass",
        question="In which of the three passes do you go through every formula, until you could build the method yourself?",
        options={
            "first": "The first",
            "second": "The second",
            "right": "The third",
        },
        correct="right",
        why={
            "first": "The first pass is five to ten minutes for the claim: title, abstract, introduction, headings and conclusion. See [Reading a paper in three passes](/course/open/res-three-pass?lang=en).",
            "second": "The second pass follows the method and the figures, and may skip a formula you do not follow yet. See [Reading a paper in three passes](/course/open/res-three-pass?lang=en).",
            "right": "This lesson is about that pass: from a formula to code you can trust.",
        },
    )
    quick_pass
    return (quick_pass,)


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="$\\mathbf{X}$ has the shape (B, 8) and $\\mathbf{W}$ the shape (8, 3). What shape does $\\mathbf{X}\\mathbf{W}$ have?",
        options={
            "right": "(B, 3)",
            "w": "(8, 3)",
            "x": "(B, 8)",
        },
        correct="right",
        why={
            "w": "The 8 is the inner size that the product sums over: it drops out. See [Gradients of matrix expressions: follow the shapes](/course/open/math-matrix-calculus?lang=en).",
            "x": "The 8 columns of X meet the 8 rows of W and drop out; the 3 columns of W stay. See [Gradients of matrix expressions: follow the shapes](/course/open/math-matrix-calculus?lang=en).",
            "right": "The outer sizes stay. This lesson gives every symbol of a formula such a shape.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A paper says what it did in formulas, and a formula is only useful once it runs. Between the two lie three traps: notation you do not know, symbols whose shape the paper never states, and code that runs but computes something slightly different. The third is the hardest to catch: it gives no error at all. A routine catches all three: decode every symbol, give it a tensor with a shape, and check the code on an example small enough to work out by hand, and against an implementation you trust.
    """)
    return


@app.cell(hide_code=True)
def predict_target():
    guess_target = L.guess(
        "right-token",
        question="The transformer paper trains with label smoothing of $\\varepsilon = 0.1$, over a vocabulary of about 37,000 tokens, and cites Szegedy and others (2016) for it. By their definition, what probability does the target give the right token?",
        options={
            "exact": "Exactly 0.9",
            "right": "Just over 0.9",
            "small": "0.1",
        },
        reveal="**Just over 0.9.** Their formula, below, gives the right token $1 - \\varepsilon + \\varepsilon / K = 0.9 + 0.1 / 37{,}000 \\approx 0.900003$, and every other token $0.1 / 37{,}000$. The authors' own code (tensor2tensor) chose a variant: exactly 0.9 for the right token, and the 0.1 spread over the other tokens only. In words, both are \"about 0.9\"; in code, you have to pick one, and only the formula or the paper's code tells you which.",
    )
    guess_target
    return (guess_target,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Step 1: decode the notation.** Label smoothing comes from Szegedy and others (2016). Instead of the one-hot target, they train against
        """),
        L.formula(r"q'(k \mid x) = (1 - \varepsilon)\,\delta_{k,y} + \varepsilon\, u(k)"),
        mo.md(r"""
        Read it symbol by symbol, with the text around it: $x$ is the input; $k$ is a class, from 1 to $K$ (in code from 0 to $K - 1$); $y$ is the right class of $x$; $\delta_{k,y}$, the **Kronecker delta**, is 1 when $k = y$ and 0 otherwise, so it is the one-hot target written as a formula; $u(k)$ is a distribution over the classes, in the paper the uniform one, $1/K$. And $\varepsilon$ is how much of the target goes to $u$: not a small number against dividing by zero, nor noise. The same letter can mean something else in another paper; the text says which. The loss is the cross-entropy against that target, with $p(k)$ the model's probabilities:
        """),
        L.formula(r"H(q', p) = -\sum_{k=1}^{K} q'(k) \log p(k)"),
        mo.md(r"""
        The picture: the target of one example with 5 classes and $\varepsilon = 0.2$, against the one-hot target.
        """),
        mo.Html(C.targets_picture({"title": "One-hot or smoothed", "x": "class k", "y": "target", "right": "0.84", "rest": "0.04",
                                   "hard": "one-hot", "soft": "smoothed", "lang": L.lang,
                                   "alt": "Bars for 5 classes, the right class is class 2. The one-hot target is 1 for class 2 and 0 for the others. The smoothed target with epsilon 0.2 is 0.84 for class 2 and 0.04 for each of the others."})),
        mo.md(r"""
        **Step 2: a tensor and a shape for every symbol.** The paper writes one example; code works on a batch of $B$ examples.

        | Symbol | Tensor | Shape |
        |---|---|---|
        | $p(k)$ | `p = torch.softmax(z, dim=1)` | (B, K) |
        | $y$ | `y` | (B,) |
        | $\delta_{k,y}$ | `torch.eye(K)[y]` | (B, K) |
        | $u(k) = 1/K$ | `1 / K` | one number |
        | $q'(k)$ | `q` | (B, K) |
        | $\sum_k$ | `.sum(dim=1)` | (B, K) to (B,) |

        One thing the formula leaves open: the sum or the mean over the batch? PyTorch takes the mean.

        **Step 3: a toy example you can work out by hand.** With $K = 2$, $p = (0.75, 0.25)$, $y$ the first class and $\varepsilon = 0.2$: $q' = (0.8 + 0.1,\ 0.1) = (0.9, 0.1)$, and $H = -(0.9 \log 0.75 + 0.1 \log 0.25) \approx 0.3975$. The logits $z = (\log 3, 0)$ give exactly that $p$. Your code must give 0.3975 too.

        **Step 4: a check against an implementation you trust.** PyTorch has this loss: `nn.CrossEntropyLoss(label_smoothing=0.2)`. Compare on random inputs with `torch.allclose`, which is true when two tensors agree up to rounding. Where there is no implementation, check special cases instead: with $\varepsilon = 0$, the loss must be the plain cross-entropy.

        ### New words and symbols

        | Word or symbol | What it means |
        |---|---|
        | **Kronecker delta** $\delta_{k,y}$ | 1 when $k = y$, 0 otherwise; in code `torch.eye(K)[y]` |
        | **label smoothing** | training against a target that gives the right class $1 - \varepsilon + \varepsilon/K$ and every other class $\varepsilon/K$ |
        | **toy example** | inputs so small that you can work out the right output by hand |
        | `torch.allclose(a, b)` | true when two tensors agree up to rounding |
        """),
        L.watch_out("A formula in a paper is usually for one example, and it leaves open what code must decide: the mean or the sum over a batch, which base the log has, which tokens count. Look for the paper's code, or check against an implementation, before you trust your own."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To turn a formula from a paper into code:

    1. Decode: write down what every symbol means, from the text around the formula, and note each one that means something other than what you are used to.
    2. Shapes: give every symbol a tensor with a shape, add the batch axis, and note what the formula leaves open.
    3. Toy: choose inputs so small that you can work out the output by hand, and compare.
    4. Numbers: compare with an implementation you trust, or with special cases, using `torch.allclose`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    New here are the Kronecker delta as `torch.eye`, and the two checks, by hand and with `torch.allclose`. The formula is the plain cross-entropy, $-\sum_k \delta_{k,y} \log p(k)$, on a toy batch of two examples.
    """)
    return


@app.cell
def example_formula():
    import math
    import torch
    import torch.nn as nn

    _z = torch.tensor([[math.log(3), 0.0], [0.0, 0.0]])  # logits, (B, K) = (2, 2)
    _y = torch.tensor([0, 1])  # the right classes, (B,)
    _K = _z.shape[1]
    _p = torch.softmax(_z, dim=1)  # p(k), (B, K)
    _delta = torch.eye(_K)[_y]  # the Kronecker delta: the one-hot targets, (B, K)
    _loss = -(_delta * torch.log(_p)).sum(dim=1).mean()  # sum over k, then mean over the batch
    print(round(_loss.item(), 4))
    print(torch.allclose(_loss, nn.CrossEntropyLoss()(_z, _y)))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    By hand: the first example has $p = (0.75, 0.25)$ and the right class 0, so $-\log 0.75 \approx 0.2877$; the second has $p = (0.5, 0.5)$, so $-\log 0.5 \approx 0.6931$. Their mean is 0.4904, as the code prints. And `torch.allclose` says PyTorch's loss is the same: the batch mean was the right choice.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_smoothed():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Label smoothing from the formula

        Write `smoothed_loss(z, y, eps)`: the cross-entropy $H(q', p)$ against the smoothed target $q'$ of Step 1, averaged over the batch. `z` holds the logits, shape (B, K); `y` the right classes, (B,); `eps` is $\varepsilon$. The check calls `smoothed_loss(torch.tensor([[math.log(3), 0.0]]), torch.tensor([0]), 0.2)`, the toy example of Step 3, which should give 0.3975, and tries more examples at once, more classes, and $\varepsilon = 0$.
        """),
        L.task("Write `smoothed_loss`: the target q' as a tensor, then H(q', p) for every example, and its mean over the batch."),
    ])
    return


@app.cell(hide_code=True)
def exercise_smoothed():
    box_smoothed = L.code_box("smoothed-loss", lines=10)
    box_smoothed
    return (box_smoothed,)


@app.cell(hide_code=True)
def check_smoothed(box_smoothed):
    result_smoothed = L.check("smoothed-loss", box_smoothed.value)
    result_smoothed
    return (result_smoothed,)


@app.cell(hide_code=True)
def hints_smoothed():
    L.hints(
        "smoothed-loss",
        [
            "The table of Step 2 gives every symbol of the formula its tensor; the worked example shows the sum and the mean. Only the target changes: $(1 - \\varepsilon)$ times the delta, plus $\\varepsilon / K$.",
        ],
        why="`K = z.shape[1]`, `p = torch.softmax(z, dim=1)`, `q = (1 - eps) * torch.eye(K)[y] + eps / K`, and `return -(q * torch.log(p)).sum(dim=1).mean()`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_factor():
    mo.md(r"""
    ### Exercise 2. A factor of four
    """)
    return


@app.cell(hide_code=True)
def choice_factor_cell():
    choice_factor = L.choice(
        "factor-four",
        question="You compare your own label smoothing with `nn.CrossEntropyLoss(label_smoothing=0.1)` on a batch of 4 examples. Yours is exactly 4 times as big. What do you check first?",
        options={
            "eps": "Whether your ε is 4 times too big",
            "log": "Whether your log has another base",
            "right": "Whether you sum over the batch",
        },
        correct="right",
        why={
            "eps": "A bigger ε changes the target, and so the loss, but not by a clean factor that equals the batch size.",
            "log": "Another base changes every log by the same factor, but that factor is about 1.44 for base 2 and 0.43 for base 10, not 4.",
            "right": "A factor that equals the batch size points to the batch axis: the formula is for one example, and PyTorch takes the mean over the batch.",
        },
    )
    choice_factor
    return (choice_factor,)


@app.cell(hide_code=True)
def text_exercise_toy():
    mo.md(r"""
    ### Exercise 3. Why a toy example
    """)
    return


@app.cell(hide_code=True)
def explain_toy():
    explain_toy_cell = L.explain(
        "why-toy",
        question="Your code agrees with PyTorch on random inputs. Why still check it on a toy example you worked out by hand from the formula?",
        model_answer="**The implementation is not the paper:** agreeing with PyTorch shows that your code does what PyTorch does, and PyTorch may have chosen another variant or another default than the paper, such as the mean over the batch, or ε spread over the other classes only. A toy example worked out from the formula itself tests your code against the paper. **It is independent:** the hand computation shares no code with yours, so a mistake in both at once is unlikely, and when they disagree, the small numbers show where. **Often there is none:** for a new method there is no implementation to compare with, and the toy example, with special cases such as ε = 0, is all you have.",
        points=[
            "you say that an implementation may differ from the paper, so agreeing with it does not prove agreeing with the formula",
            "you say that the toy example comes from the formula itself, independent of any code, and small enough to find where it goes wrong",
            "you say that for new methods there often is no implementation to compare with",
        ],
    )
    explain_toy_cell
    return (explain_toy_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The label smoothing you found in your third pass of the transformer paper, in [Reading a paper in three passes](/course/open/res-three-pass?lang=en), is now code, with every symbol given a shape as in [Gradients of matrix expressions: follow the shapes](/course/open/math-matrix-calculus?lang=en). The next lesson, *Critical reading: claims versus evidence*, asks whether a paper's experiments really support what it claims.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_factor, explain_toy_cell, result_smoothed):
    L.wrap_up(result_smoothed, choice_factor, explain_toy_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
