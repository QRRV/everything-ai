# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-rmsnorm. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("llm-rmsnorm", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Layer normalization](/course/open/nn-layernorm?lang=en) and [Assemble your GPT](/course/open/gpt-assemble?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_ln():
    quick_ln = L.choice(
        "quick-ln",
        question="Layer normalization turns the row $[1, 2, 3]$ into $\\hat{x}$, before $\\gamma$ and $\\beta$. What do you know about $\\hat{x}$?",
        options={
            "right": "Its mean is 0 and its variance about 1",
            "sum": "Its numbers add up to 1, as after a softmax",
            "same": "It is the same row: three numbers need no normalizing",
        },
        correct="right",
        why={
            "right": "Layer normalization takes the mean of the row off and divides by its standard deviation: about $[-1.22, 0, 1.22]$.",
            "sum": "That is softmax. Layer normalization centers and rescales the row. See [Layer normalization](/course/open/nn-layernorm?lang=en).",
            "same": "Layer normalization works on any row of two numbers or more. See [Layer normalization](/course/open/nn-layernorm?lang=en).",
        },
    )
    quick_ln
    return (quick_ln,)


@app.cell(hide_code=True)
def quick_check_where():
    quick_where = L.choice(
        "quick-where",
        question="Where does your GPT of [Assemble your GPT](/course/open/gpt-assemble?lang=en) normalize?",
        options={
            "after": "After each block, on its output",
            "right": "Before attention and before the MLP in every block, and once more before the output head",
            "input": "Once, on the embeddings, before the first block",
        },
        correct="right",
        why={
            "after": "Your blocks are pre-norm: each part gets a normalized input, and adds its output to the residual stream. See [Assemble your GPT](/course/open/gpt-assemble?lang=en).",
            "right": "Two norms per block, and one at the end: 5 in a GPT of 2 blocks. RMSNorm goes in exactly those places.",
            "input": "Every block normalizes the input of its attention and of its MLP. See [Assemble your GPT](/course/open/gpt-assemble?lang=en).",
        },
    )
    quick_where
    return (quick_where,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Llama, Mistral, Qwen and Gemma do not use layer normalization. They use **RMSNorm**: layer normalization without the mean. It does less work per token, it has half the parameters, and in large language models it trains about as well. The models you will read about and the open recipes you will follow are built with it, and so is the modern GPT you build in a few lessons.
    """)
    return


@app.cell(hide_code=True)
def predict_rms():
    guess_rms = L.guess(
        "rms-of-twos",
        question="Layer normalization turns the row $[2, 2, 2, 2]$ into zeros: every number equals the mean. RMSNorm divides the row by the square root of the mean of its squares. With $\\gamma = 1$, what does RMSNorm make of $[2, 2, 2, 2]$?",
        options={
            "zeros": "$[0, 0, 0, 0]$",
            "right": "$[1, 1, 1, 1]$",
            "half": "$[0.5, 0.5, 0.5, 0.5]$",
            "same": "$[2, 2, 2, 2]$",
        },
        reveal="**$[1, 1, 1, 1]$.** The mean of the squares is 4, its square root 2, and $2 / 2 = 1$. RMSNorm keeps the direction of the row and only sets its size: a row whose numbers are all alike stays a row whose numbers are all alike.",
    )
    guess_rms
    return (guess_rms,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Step 1: only rescale.** The root mean square of a row, its RMS, is the square root of the mean of its squares. RMSNorm divides every number of the row by it, and multiplies by a learned scale $\gamma_j$ per feature:
        """),
        L.formula(r"\begin{aligned} \text{RMS}(x) &= \sqrt{\tfrac{1}{d} \textstyle\sum_k x_k^2 + [[hyper:\varepsilon]]} \\ y_j &= [[param:\gamma_j]] \, \frac{x_j}{\text{RMS}(x)} \end{aligned}"),
        mo.md(r"""
        As in layer normalization, $\varepsilon$ keeps a row of zeros from a division by 0, and $\gamma$ starts at 1. There is no mean to take off, and no $\beta$.

        **Step 2: what it keeps, and what it drops.** Like layer normalization, RMSNorm does not care about the size of its input: $3x$ gives the same as $x$. That is what keeps the numbers in a deep network at a steady size. Unlike layer normalization, it does care about a shift: $x + 5$ gives something other than $x$. The paper that introduced RMSNorm argued that the rescaling is what matters and that the centering can go, and in its experiments models trained about as well without it.

        **Step 3: what it saves.** Per row, layer normalization computes two numbers, the mean and the variance, and takes the mean off every feature; RMSNorm computes one, the mean of the squares, and takes nothing off. Per norm, layer normalization has $2d$ parameters, RMSNorm $d$. Whether that makes it faster depends on the code: PyTorch's `nn.LayerNorm` runs as one fused kernel, a single program that does the whole normalization in one go, and on a CPU its `nn.RMSNorm` does not, so there layer normalization can even be faster. Large models use a fused RMSNorm.

        **Step 4: where it goes.** In exactly the places of layer normalization: before attention and before the MLP in every block, and once before the output head. PyTorch has it as `nn.RMSNorm(d)`.

        The picture: one row, $x = [3, 5, 4, 6, 5, 7]$, and what layer normalization and RMSNorm make of it.
        """),
        mo.Html(C.norms_picture({"title": "One row, normalized twice", "x": "feature", "y": "value", "input": "x", "ln": "LayerNorm", "rms": "RMSNorm", "lang": L.lang,
                                 "alt": "Six features. The input x goes from 3 to 7. Layer normalization, open dots, centers it around 0, from about −1.5 to 1.5. RMSNorm, filled dots, keeps it above 0 and divides it by its RMS of about 5.2: from about 0.6 to 1.4, the same shape as x."})),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **RMS** | root mean square: the square root of the mean of the squares | |
        | **RMSNorm** | dividing each row by its RMS, then a learned scale per feature | `nn.RMSNorm(d)` |
        """),
        L.watch_out("RMSNorm of a row is not the same as layer normalization of it, even when $\\gamma$ is the same: only for a row whose mean is 0 do the two agree. A trained model's weights belong to the norm it was trained with."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To normalize a row with RMSNorm:

    1. Square every number, and take the mean along the last axis, keeping that axis.
    2. Add $\varepsilon$, and take the square root: the RMS of the row.
    3. Divide the row by its RMS, and multiply each feature by its $\gamma$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    PyTorch's `nn.RMSNorm` is new: here it runs next to `nn.LayerNorm` on two rows of 6 numbers, once rescaled and once shifted.
    """)
    return


@app.cell
def example_invariance():
    import torch

    torch.manual_seed(0)
    _x = torch.randn(2, 6)
    _rms = torch.nn.RMSNorm(6, eps=1e-6)
    _ln = torch.nn.LayerNorm(6)
    print("RMSNorm, 3x the same as x:", torch.allclose(_rms(3 * _x), _rms(_x), atol=1e-5))
    print("RMSNorm, x + 5 the same as x:", torch.allclose(_rms(_x + 5), _rms(_x), atol=1e-5))
    print("layer normalization, x + 5 the same as x:", torch.allclose(_ln(_x + 5), _ln(_x), atol=1e-5))
    print("parameters:", sum(p.numel() for p in _rms.parameters()), "against", sum(p.numel() for p in _ln.parameters()))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Rescaled, RMSNorm gives the same; shifted, it does not, while layer normalization does not notice the shift. RMSNorm has 6 parameters, one $\gamma$ per feature; layer normalization has 12, with a $\beta$ per feature as well.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rms():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Your own RMSNorm

        Write the class `RMSNorm(d, eps=1e-6)`: a weight `self.weight` of `d` numbers, $\gamma$, that starts at 1, and a `forward` that normalizes the last axis of its input, of any shape. The check runs `RMSNorm(2)` on `torch.tensor([[3.0, 4.0], [1.0, 1.0]])`, which should give about `[[0.8485, 1.1314], [1.0, 1.0]]`, and tries a mini-batch of tokens, another weight and a large eps. The weight must be an `nn.Parameter`, as the `.weight` of an `nn.Linear` is: `nn.Parameter(t)` makes the tensor `t` a parameter of the module, one that `parameters()` finds.
        """),
        L.task("Write `RMSNorm`: a weight that starts at 1, and a `forward` that divides each row by its RMS and scales it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rms():
    box_rms = L.code_box("rmsnorm", lines=13)
    box_rms
    return (box_rms,)


@app.cell(hide_code=True)
def check_rms(box_rms):
    result_rms = L.check("rmsnorm", box_rms.value)
    result_rms
    return (result_rms,)


@app.cell(hide_code=True)
def hints_rms():
    L.hints(
        "rmsnorm",
        [
            "The recipe, with `x * x`, `.mean(dim=-1, keepdim=True)` and `torch.sqrt`; the weight as an `nn.Parameter` around `torch.ones(d)`.",
        ],
        why="`keepdim=True` keeps one RMS per row as a column, so that every row is divided by its own; eps goes inside the square root; the weight, `torch.ones(d)`, makes a new norm a pure rescaling.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### LayerNorm against your RMSNorm

    The button trains your `GPT` from [Assemble your GPT](/course/open/gpt-assemble?lang=en) twice on Tiny Shakespeare, with the same start and the same 400 mini-batches: once as it is, and once with your `RMSNorm` in the place of every `nn.LayerNorm`. A small GPT, 2 blocks of $d = 64$, on the CPU: it takes about a minute. Where a module of yours is missing, the course's version stands in. It works once exercise 1 is right.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train with LayerNorm and with RMSNorm")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_train(box_rms, result_rms, train_button):
    if result_rms.state != "correct":
        _out = mo.md("*Finish exercise 1 first: this cell uses your `RMSNorm`.*")
    elif not train_button.value:
        _out = mo.md("*Press the button to train.*")
    else:
        _m = C.demo(box_rms.value)
        if _m == "error":
            _out = mo.md("*The training stopped with an error: does your GPT of [Assemble your GPT](/course/open/gpt-assemble?lang=en) still work?*")
        elif _m == "norms":
            _out = mo.md("*Your GPT has no `nn.LayerNorm` to swap: does it use `nn.LayerNorm`, as in [Assemble your GPT](/course/open/gpt-assemble?lang=en)?*")
        else:
            _n = lambda _x, _d=0: C.number(_x, _d, L.lang)
            _l, _r = _m["layer"], _m["rms"]
            _stand = (" The course's version stood in for: " + ", ".join(f"`{_s}`" for _s in _m["stand_ins"]) + ".") if _m["stand_ins"] else ""
            _out = mo.md(f"""
    | | LayerNorm | your RMSNorm |
    |---|---|---|
    | validation loss | {_n(_l['val_loss'], 3)} | {_n(_r['val_loss'], 3)} |
    | parameters | {_n(_l['params'])} | {_n(_r['params'])} |
    | seconds | {_n(_l['seconds'], 1)} | {_n(_r['seconds'], 1)} |

    Your RMSNorm replaced {_r['swapped']} normalization layers, and the model lost {_n(_l['params'] - _r['params'])} parameters with them. In the course's own runs with seeds 0, 1 and 2, RMSNorm ended at most 0.014 lower and at most 0.001 higher, while the seed alone moved layer normalization's loss by more than 0.02: a difference of that size is noise. The seconds differ little: the norms are only a small part of the work, and a norm you wrote yourself, in a few PyTorch operations, is no faster than the fused `nn.LayerNorm`.{_stand}
    """)
    _out
    return


@app.cell(hide_code=True)
def text_exercise_betas():
    mo.md(r"""
    ### Exercise 2. What a GPT saves
    """)
    return


@app.cell(hide_code=True)
def choice_betas_cell():
    choice_betas = L.choice(
        "betas",
        question="A GPT with the sizes of GPT-2 small, 12 blocks of $d = 768$, has 25 normalization layers: two per block and one at the end. You replace them all by RMSNorm. How many parameters does it lose?",
        options={
            "both": "38,400: the $\\gamma$ and $\\beta$ of every norm",
            "none": "None: the norms keep their size",
            "right": "19,200: the $\\beta$ of every norm",
        },
        correct="right",
        why={
            "both": "RMSNorm keeps its $\\gamma$, one per feature: only $\\beta$ goes.",
            "none": "Every layer normalization has a $\\beta$ of $d$ numbers that RMSNorm does not have.",
            "right": "$25 \\times 768 = 19{,}200$, out of about 124 million: RMSNorm saves work, not parameters that matter.",
        },
    )
    choice_betas
    return (choice_betas,)


@app.cell(hide_code=True)
def text_exercise_mean():
    mo.md(r"""
    ### Exercise 3. Without the mean
    """)
    return


@app.cell(hide_code=True)
def explain_no_mean():
    explain_mean_cell = L.explain(
        "why-no-mean",
        question="RMSNorm leaves out what layer normalization does first: taking the mean off. Why does a model lose little by that, and what does it gain?",
        model_answer="**What stays:** the division by the RMS still holds every token at a steady size, whatever the size of the numbers that come in; that rescaling is what keeps a deep network stable. **Why the mean can go:** the steady size comes from the rescaling, not from the mean of 0; the paper that introduced RMSNorm found that models train about as well without the centering, and today's large language models train well with RMSNorm. **What it gains:** one number to compute per row instead of two, nothing to take off, and no $\\beta$: less work per token, which pays off in a fused kernel.",
        points=[
            "you say that the rescaling, which keeps every token at a steady size, stays",
            "you give a reason why the centering adds little, such as that the rescaling alone gives the steady size, or that models train about as well without it",
            "you name the gain: one number per row instead of two, and no beta",
        ],
    )
    explain_mean_cell
    return (explain_mean_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The layer normalization of [Layer normalization](/course/open/nn-layernorm?lang=en), with its mean and its $\beta$ left out, in the places it has in [Assemble your GPT](/course/open/gpt-assemble?lang=en). The next lesson, *Modern MLP blocks: GELU, SiLU and SwiGLU*, changes another part of every block: its MLP.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_betas, explain_mean_cell, result_rms):
    L.wrap_up(result_rms, choice_betas, explain_mean_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
