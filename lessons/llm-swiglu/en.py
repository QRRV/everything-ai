# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-swiglu. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("llm-swiglu", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [The transformer block](/course/open/attn-transformer-block?lang=en) and [Assemble your GPT](/course/open/gpt-assemble?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mlp():
    quick_mlp = L.choice(
        "quick-mlp",
        question="In a block of your GPT, which part works on each token by itself, without looking at the other tokens?",
        options={
            "attention": "The attention: each token looks only at itself",
            "right": "The MLP: the same layers, token by token",
            "both": "Both: every part of a block works per token",
        },
        correct="right",
        why={
            "attention": "Attention is where tokens look at each other, at all the tokens before them. See [The transformer block](/course/open/attn-transformer-block?lang=en).",
            "right": "Attention mixes the tokens; the MLP then transforms each token's vector by itself. This lesson changes that MLP.",
            "both": "Attention mixes information between the tokens; only the MLP and the normalizations work per token. See [The transformer block](/course/open/attn-transformer-block?lang=en).",
        },
    )
    quick_mlp
    return (quick_mlp,)


@app.cell(hide_code=True)
def quick_check_relu():
    quick_relu = L.choice(
        "quick-relu",
        question="What does ReLU give for an input of $-2$, and for an input of $3$?",
        options={
            "abs": "2 and 3",
            "neg": "$-2$ and 3",
            "right": "0 and 3",
        },
        correct="right",
        why={
            "abs": "ReLU does not flip negative numbers: it sets them to 0. See [A hidden layer and ReLU](/course/open/nn-hidden-layer?lang=en).",
            "neg": "That is the input itself. ReLU keeps positive numbers and sets negative ones to 0: see [A hidden layer and ReLU](/course/open/nn-hidden-layer?lang=en).",
            "right": "And its slope is 0 for every negative input: there, no gradient gets through.",
        },
    )
    quick_relu
    return (quick_relu,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The MLP of your GPT is two layers with a ReLU between them, as in the transformer of 2017. GPT-2 already used GELU instead of ReLU, and Llama, Mistral and PaLM use **SwiGLU**: a smooth activation function and a gate, an extra layer whose output multiplies the hidden numbers, number by number. The MLP holds about two thirds of the parameters of a large model, so how it is built matters.
    """)
    return


@app.cell(hide_code=True)
def predict_size():
    guess_size = L.guess(
        "llama-hidden",
        question="Llama 2 7B has vectors of $d = 4096$ numbers. A classic MLP would widen them to $4d = 16{,}384$; its SwiGLU widens them to 11,008. Why that number?",
        options={
            "memory": "It is the largest width whose weights fit in the memory of one GPU",
            "slow": "Its smooth activation function takes longer than ReLU; a smaller width makes up for that time",
            "right": "Three weight matrices instead of two; a smaller width keeps the parameter count",
        },
        reveal="**Three matrices instead of two.** A classic MLP has $2 \\times 4096 \\times 16{,}384$ weights; SwiGLU has $3 \\times 4096$ times its hidden width. The two are equal at a width of $\\frac{8}{3} \\times 4096 \\approx 10{,}923$, which Llama rounds up to a multiple of 256: 11,008. The smooth activation function itself costs only a few operations per hidden number, little next to the matrices.",
    )
    guess_size
    return (guess_size,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Step 1: smooth activation functions.** **GELU** multiplies $x$ by the probability that a number from the normal distribution with mean 0 and standard deviation 1 is below $x$; **SiLU** multiplies $x$ by its sigmoid:
        """),
        L.formula(r"\text{SiLU}(x) = x \, \sigma(x) = \frac{x}{1 + e^{-x}}"),
        mo.md(r"""
        Both are close to $x$ for large $x$ and close to 0 for very negative $x$, like ReLU. Unlike ReLU they have no corner at 0, they dip a little below 0, SiLU to about $-0.28$, and their slope for a negative input is not 0: a small gradient still gets through.

        **Step 2: a gate.** A gated MLP, or **gated linear unit** (GLU), has two layers from $d$ to the hidden width instead of one. One, the gate, goes through SiLU; the other, up, does not; their outputs are multiplied number by number; and a third layer, down, takes the result back to $d$:
        """),
        L.formula(r"\begin{gathered} \mathbf{h} = \text{SiLU}(\mathbf{x} [[param:\mathbf{W}_{\text{gate}}]]) \odot \mathbf{x} [[param:\mathbf{W}_{\text{up}}]] \\ \text{SwiGLU}(\mathbf{x}) = \mathbf{h} \, [[param:\mathbf{W}_{\text{down}}]] \end{gathered}"),
        mo.md(r"""
        Each hidden number is now a product: the gate decides, from the input, how much of the up value passes, and even its sign. Llama's layers have no bias.

        **Step 3: the same parameters.** A classic MLP with a hidden width of $4d$ has $2 \cdot d \cdot 4d = 8d^2$ weights. SwiGLU has three matrices of $d$ by its hidden width; with a width of $\frac{8}{3} d$ that is $3 \cdot d \cdot \frac{8}{3} d = 8d^2$ as well, and so are the FLOPs of the matrix products: SwiGLU replaces the classic MLP at about the same cost.

        **Step 4: why it helps.** In the paper that compared gated MLPs in transformers, the gated versions, SwiGLU among them, reached a lower loss than ReLU and GELU at the same number of parameters and FLOPs. The paper offered no explanation; the smooth slope and the product of two projections are the usual guesses.

        The picture: ReLU, dashed, GELU, thin, and SiLU, thick, from $-4$ to $3$.
        """),
        mo.Html(C.activations_picture({"title": "Three activation functions", "y": "output", "dip": "dips below 0", "lang": L.lang,
                                       "alt": "ReLU, dashed, is 0 for every negative x and x from 0 on, with a corner at 0. GELU, thin, and SiLU, thick, follow it for large x, but are smooth: they dip a little below 0 for negative x, GELU to about −0.17, SiLU to about −0.28, and slowly return to 0."})),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **GELU** | $x$ times the probability that a number from the normal distribution with mean 0 and standard deviation 1 is below $x$ | `F.gelu(x)` |
        | **SiLU** | $x$ times its sigmoid; also called swish | `F.silu(x)` |
        | **gated linear unit** (GLU) | two linear layers on the same input, multiplied number by number; one, the gate, goes through an activation function first | |
        | **SwiGLU** | a GLU with SiLU on the gate: SiLU of the gate times up, then down | |
        """),
        L.watch_out("GELU has two versions: the exact one, and a faster approximation with tanh that GPT-2 used. `F.gelu(x)` gives the exact one, `F.gelu(x, approximate=\"tanh\")` the other: a model trained with one gives slightly different outputs with the other."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To build a SwiGLU MLP for vectors of $d$ numbers:

    1. Pick the hidden width: about $\frac{8}{3}d$ for the parameters of a classic MLP of $4d$.
    2. Make three linear layers without bias: gate and up from $d$ to the hidden width, down from the hidden width back to $d$.
    3. Take SiLU of the gate, multiply it number by number by up, and send the product through down.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    GELU and SiLU in PyTorch are new: ReLU, GELU and SiLU at a few inputs, and their slopes, by autograd.
    """)
    return


@app.cell
def example_activations():
    import torch
    import torch.nn.functional as F

    _x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    for _name, _f in [("ReLU", torch.relu), ("GELU", F.gelu), ("SiLU", F.silu)]:
        _y = _f(_x)
        _y.sum().backward()
        print(_name, "values:", [round(_v, 3) for _v in _y.tolist()])
        print(_name, "slopes:", [round(_v, 3) for _v in _x.grad.tolist()])
        _x.grad = None
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    At $x = -1$, ReLU gives 0 with slope 0; GELU gives $-0.159$ with slope $-0.083$, and SiLU $-0.269$ with slope $0.072$: a weight that feeds them still gets a gradient. At $x = 2$, GELU is close to 2, at 1.954, and SiLU lower, at 1.762: the larger $x$, the closer they get. At $x = 0$, PyTorch gives ReLU the slope 0, while GELU and SiLU have the slope 0.5 there.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_swiglu():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Your own SwiGLU

        Write the class `SwiGLU(d, hidden)` with three linear layers without bias, named `self.gate`, `self.up` and `self.down`, and a `forward` that multiplies SiLU of the gate by up, number by number, and sends the product through down. The check sets the weights by hand: `SwiGLU(2, 3)` with the weights its message shows, on the token `[1.0, -1.0]`, should give about `[[1.4621, 0.5379]]`. It also counts the parameters and runs a mini-batch of tokens.
        """),
        L.task("Write `SwiGLU`: `self.gate`, `self.up` and `self.down` without bias, and a `forward` that multiplies the up values by SiLU of the gate."),
    ])
    return


@app.cell(hide_code=True)
def exercise_swiglu():
    box_swiglu = L.code_box("swiglu", lines=14)
    box_swiglu
    return (box_swiglu,)


@app.cell(hide_code=True)
def check_swiglu(box_swiglu):
    result_swiglu = L.check("swiglu", box_swiglu.value)
    result_swiglu
    return (result_swiglu,)


@app.cell(hide_code=True)
def hints_swiglu():
    L.hints(
        "swiglu",
        [
            "The recipe, steps 2 and 3: `nn.Linear(..., bias=False)` three times, and in `forward` one line with `F.silu` and `*`.",
        ],
        why="Only the gate goes through SiLU; `*` multiplies number by number; `self.down` takes the `hidden` numbers back to `d`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_hidden():
    mo.md(r"""
    ### Exercise 2. The same parameters
    """)
    return


@app.cell(hide_code=True)
def choice_hidden_cell():
    choice_hidden = L.choice(
        "hidden-size",
        question="A GPT with $d = 768$ has classic MLPs that widen to $4d = 3{,}072$. You replace them by SwiGLU with the same number of parameters. What hidden width do you give it?",
        options={
            "right": "2,048, two thirds of the width",
            "same": "3,072, the same width",
            "half": "1,536, half the width",
        },
        correct="right",
        why={
            "same": "With three matrices of 768 by 3,072, SwiGLU would have half as many parameters again as the classic MLP.",
            "half": "Then SwiGLU has $3 \\times 768 \\times 1{,}536$ weights, a quarter fewer than the classic MLP's $2 \\times 768 \\times 3{,}072$.",
            "right": "$\\frac{8}{3} \\times 768 = 2{,}048$: $3 \\times 768 \\times 2{,}048 = 2 \\times 768 \\times 3{,}072$.",
        },
    )
    choice_hidden
    return (choice_hidden,)


@app.cell(hide_code=True)
def text_exercise_gate():
    mo.md(r"""
    ### Exercise 3. What the gate adds
    """)
    return


@app.cell(hide_code=True)
def explain_gate():
    explain_gate_cell = L.explain(
        "what-the-gate-adds",
        question="In SwiGLU, every hidden number is SiLU of a gate times an up value. What can this MLP do that the ReLU MLP of your GPT cannot, and what does it cost at the same number of parameters?",
        model_answer="**What it can do:** each hidden number is a product of two projections of the input, so the input decides, unit by unit, how much of a value passes, and with which sign; a ReLU unit can only cut its own value off below 0. And SiLU has a slope for negative inputs, so the gate's weights still get a gradient there. **What it costs:** a third weight matrix, so at the same parameters the hidden width shrinks to about $\\frac{8}{3}d$ instead of $4d$; the FLOPs per token stay the same, and there is one more multiplication per hidden number.",
        points=[
            "you say that each hidden number is a product, so the input controls how much of a value passes",
            "you name the slope of SiLU for negative inputs, or that ReLU cuts off everything below 0",
            "you name the cost: a third matrix, and so a smaller hidden width at the same parameters",
        ],
    )
    explain_gate_cell
    return (explain_gate_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The MLP of [The transformer block](/course/open/attn-transformer-block?lang=en), with SiLU, built on the sigmoid, in the place of ReLU, and a gate that multiplies number by number. The next lesson, *Rotary position embeddings (RoPE)*, changes how a GPT knows where a token is; in *Project: upgrade your GPT to a modern LLM* your GPT gets SwiGLU in every block.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_hidden, explain_gate_cell, result_swiglu):
    L.wrap_up(result_swiglu, choice_hidden, explain_gate_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
