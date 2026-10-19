# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson torch-nn-module. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("torch-nn-module", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en) and [Special methods: making + and print work on your objects](/course/open/py-special-methods?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_grad():
    quick_grad = L.choice(
        "quick-grad",
        question="`w = torch.tensor([1.0, 2.0], requires_grad=True)`, `loss = (w * w).sum()`, and `loss.backward()`. What is `w.grad`?",
        options={
            "sum": "`tensor(6.)`",
            "right": "`tensor([2., 4.])`",
            "none": "`None`, because `loss` is not a leaf tensor",
        },
        correct="right",
        why={
            "sum": "`.grad` has the shape of `w`: one slope per weight. See *The backward pass* in [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en).",
            "none": "`w` is the leaf tensor here, made with `requires_grad=True`, so it gets a `.grad`. See [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en).",
            "right": "The slope of $w_1^2 + w_2^2$ is $2w$ for each weight: 2 and 4, in the shape of `w`.",
        },
    )
    quick_grad
    return (quick_grad,)


@app.cell(hide_code=True)
def quick_check_call():
    quick_call = L.choice(
        "quick-call",
        question="What does this program show?\n\n```python\nclass Adder:\n    def __init__(self):\n        self.total = 0\n\n    def __call__(self, x):\n        self.total = self.total + x\n        return self.total\n\n\nadd = Adder()\nadd(4)\nprint(add(5))\n```",
        options={
            "five": "`5`",
            "error": "Nothing: Python stops, because `add` is an object, not a function",
            "right": "`9`",
        },
        correct="right",
        why={
            "five": "Each call adds to `self.total`, which the object keeps between calls. See [Special methods: making + and print work on your objects](/course/open/py-special-methods?lang=en).",
            "error": "An object whose class has `__call__` can be called like a function. See [Special methods: making + and print work on your objects](/course/open/py-special-methods?lang=en).",
            "right": "`add(4)` runs `add.__call__(4)`, and `self.total` becomes 4; `add(5)` adds 5 to it and gives back 9.",
        },
    )
    quick_call
    return (quick_call,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en) your model was two loose tensors, `w` and `b`, and the training step named each one. A network with three layers has six such tensors, and a language model hundreds. Your own project already solved that: `Layer` and `MLP` kept their parameters together, and `parameters()` gave them all at once. PyTorch does the same with **modules**. `nn.Linear` is a layer, and your own class built on `nn.Module` is a model that knows its layers, its parameters and its forward pass.

    **An everyday example.** A toolbox: every tool has its place in it. When you move, you carry the box, not every screwdriver on its own, and when you need something, you know where to look.
    """)
    return


@app.cell(hide_code=True)
def predict_weight():
    guess_weight = L.guess(
        "weight",
        question="`layer = nn.Linear(3, 2)` is a layer with 3 inputs and 2 neurons. What do you think the shape of `layer.weight` is?",
        options={
            "inputs": "(3, 2): a row of 2 weights per input",
            "flat": "(6,): all six weights in one row",
            "right": "(2, 3): a row per neuron",
        },
        reveal="**(2, 3).** `nn.Linear` keeps the weights of each neuron as a row, as `W` in [The forward pass in NumPy, with shapes](/course/open/nn-forward-numpy?lang=en). So for examples in the rows of $\\mathbf{X}$, it works out $\\mathbf{X}\\mathbf{W}^\\top + \\mathbf{b}$: the transpose from [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en). Its `bias` has shape (2,), one bias term per neuron.",
    )
    guess_weight
    return (guess_weight,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        In PyTorch, a **module** is a piece of a network: an object that holds parameters and knows its forward pass. They all build on the class `nn.Module`, which you get with `from torch import nn`.

        **A layer: `nn.Linear`.** `nn.Linear(n_inputs, n_outputs)` makes a layer of `n_outputs` neurons, each with `n_inputs` weights and a bias term. Its `.weight` has shape (outputs, inputs), one row per neuron, and its `.bias` shape (outputs,). Both are an **`nn.Parameter`**: a tensor with `requires_grad=True` that an `nn.Module` keeps track of. They start at random numbers, as the weights of your own `Neuron` did; `torch.manual_seed(0)` gives the same numbers on every run. Calling the layer on examples in the rows of `X` works out:
        """),
        L.formula(r"\mathbf{X}[[param:\mathbf{W}]]^\top + [[param:\mathbf{b}]]"),
        mo.md(r"""
        In code, `layer(X)` gives the same as `X @ layer.weight.T + layer.bias`. That is the **convention** of PyTorch: weights as rows, and examples as rows too, so the weights are transposed.

        **Your own model: a class on `nn.Module`.** Write a class that builds on `nn.Module`, as in [Building on another class](/course/open/py-inheritance?lang=en). Its `__init__` first calls `super().__init__()`, so that `nn.Module` can set itself up, and then makes each layer as an attribute: `self.hidden = nn.Linear(2, 3)`. Its method **`forward(self, x)`** says what the network does with `x`: the layers one after the other, with `torch.relu` in between. You call the model itself, `net(X)`: `nn.Module` has a `__call__`, as in [Special methods: making + and print work on your objects](/course/open/py-special-methods?lang=en), and that runs your `forward`.

        **All parameters at once.** An `nn.Module` finds every layer you gave it as an attribute, and every parameter in them: `net.parameters()` gives them all, as `parameters()` of your own `MLP` did. One thing differs: your `MLP` gave one `Value` per number, and `net.parameters()` gives whole tensors, the `weight` and the `bias` of each layer. For a network of two layers that is 4 tensors. A training step goes through them in a loop, and `p -= lr * p.grad` moves every number of a tensor at once. Only attributes count: a layer in a plain list, the way your `MLP` kept its layers, is not found.

        | In your project | In PyTorch |
        |---|---|
        | `Module` | `nn.Module` |
        | `Layer(2, 3, relu=False)` | `nn.Linear(2, 3)`: no ReLU inside |
        | the ReLU inside each `Neuron` | `torch.relu` in `forward` |
        | `MLP(2, [3, 1])` | your own class on `nn.Module`, with `forward` |
        | `m.parameters()`: one `Value` per number | `net.parameters()`: one tensor per `weight` and `bias` |
        | `m(x)`, through `__call__` | `net(X)`, through `__call__` and `forward` |

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **module** | | in PyTorch: a piece of a network, an object that holds parameters and knows its forward pass | `nn.Module` |
        | `nn.Linear` | n n linear | a layer: `nn.Linear(n_inputs, n_outputs)` works out $\mathbf{X}\mathbf{W}^\top + \mathbf{b}$ | `nn.Linear(2, 3)` |
        | `nn.Parameter` | | a tensor with `requires_grad=True` that an `nn.Module` keeps track of, such as `.weight` and `.bias` | `layer.weight` |
        | `forward` | | the method of your `nn.Module` that says what it does with its input; `net(X)` runs it | `def forward(self, x):` |
        | **convention** $\mathbf{X}\mathbf{W}^\top + \mathbf{b}$ | | PyTorch keeps the weights of each neuron as a row, and the examples as rows, so it transposes the weights | `X @ layer.weight.T + layer.bias` |
        | `torch.manual_seed` | | the seed of PyTorch's random numbers: the same starting weights on every run | `torch.manual_seed(0)` |
        """),
        L.watch_out("Module here does not mean a Python module, a file you import, as in [Your own module: keeping project code in a file](/course/open/py-own-modules?lang=en). In PyTorch, an `nn.Module` is a piece of a network: it plays the part of the class `Module` that your `Neuron`, `Layer` and `MLP` built on. You can write an `nn.Module` in a Python module, so read which one is meant."),
        mo.md(r"""
        The network of the exercises, with 2 inputs, a hidden layer `self.hidden` of 3 neurons with ReLU, and an output layer `self.out` of 1 neuron:
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_net():
    C.picture(["2 inputs", "nn.Linear(2, 3)", "nn.Linear(3, 1)"],
              title="Net: two nn.Linear layers",
              alt="Two inputs, each connected to the three neurons of the hidden layer, nn.Linear(2, 3), with ReLU; those three connected to the one output neuron, nn.Linear(3, 1).")
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To build a network as an `nn.Module`, and train it:

    1. Write a class on `nn.Module`. In `__init__`, first `super().__init__()`, then each layer as an attribute: `self.hidden = nn.Linear(2, 3)`.
    2. Write `forward(self, x)`: the layers one after the other, with `torch.relu` in between.
    3. Call the model on all examples at once: `y_pred = net(X)`.
    4. Train with `net.parameters()`: every `.grad` to `None`, `loss.backward()`, and the step for every parameter inside `with torch.no_grad():`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A network for the four corners of XOR, from [What one neuron cannot learn](/course/open/nn-xor?lang=en), with 2 inputs, a hidden layer of 4 neurons and 1 output. It is trained for 300 steps. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_module():
    import torch
    from torch import nn

    # 1. the class, with its layers as attributes
    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(2, 4)
            self.out = nn.Linear(4, 1)

        # 2. the forward pass
        def forward(self, x):
            return self.out(torch.relu(self.hidden(x)))

    torch.manual_seed(1)
    _net = _Net()
    print(_net)
    # 3. all examples at once: (4, 2) -> (4, 1)
    _X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    _y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    print(_net(_X))
    # 4. training, with every parameter of the network
    for _p in _net.parameters():
        print(_p.shape)
    for _step in range(300):
        _loss = ((_net(_X) - _y) ** 2).mean()
        for _p in _net.parameters():
            _p.grad = None
        _loss.backward()
        with torch.no_grad():
            for _p in _net.parameters():
                _p -= 0.1 * _p.grad
    print(_loss.item())
    print(_net(_X))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `print(_net)` shows the two layers by their attribute names; marimo writes the class `_Net` as `_cell_…_Net`, because a name with `_` belongs to its own cell. Before training, `_net(_X)` gives four numbers that have nothing to do with XOR. `_net.parameters()` gives four tensors, in the order the layers were made: the `weight` of the hidden layer, shape (4, 2), a row of 2 weights for each of its 4 neurons; its `bias`, shape (4,); and the `weight` (1, 4) and `bias` (1,) of the output layer. Together they hold 4 · 2 + 4 + 4 + 1 = 17 numbers, and the loop moves all of them without naming one. After 300 steps the loss is almost 0, and the outputs are almost 0, 1, 1 and 0. PyTorch writes them short, as in [Too small, too big: the learning rate](/course/open/ml-learning-rate?lang=en): `9.9943e-01` is 0.99943, and `1.9103e-03` is 0.0019103.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. What comes out of a layer
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "output-shape",
        question="`layer = nn.Linear(5, 3)`, and `X` holds 10 examples with 5 inputs each: shape (10, 5). What is the shape of `layer(X)`?",
        options={
            "right": "(10, 3)",
            "weight": "(3, 5)",
            "inputs": "(10, 5)",
        },
        correct="right",
        why={
            "weight": "That is the shape of `layer.weight`, one row per neuron. What does the layer give for each example? See the formula in *The idea*.",
            "inputs": "Each example goes in with 5 inputs; how many numbers come out, one per what? See the formula in *The idea*.",
            "right": "$\\mathbf{X}\\mathbf{W}^\\top$ is (10, 5) @ (5, 3), and the bias term of each neuron is added to every row: one row per example, one column per neuron.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_net():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Your own nn.Module

        Finish the class `Net`: a network with 2 inputs, a hidden layer of 3 neurons with ReLU, and an output layer of 1 neuron, as in the picture. Replace the two `...`. Make the layers in the order of the network, `self.hidden` first and then `self.out`: PyTorch draws their random starting weights in the order you make them. The last lines make the network with seed 0 and call it on the four corners of XOR.
        """),
        L.task("Make `Net` an `nn.Module` with the layers `self.hidden` and `self.out`, and a `forward` with ReLU between them."),
    ])
    return


@app.cell(hide_code=True)
def exercise_net():
    box_net = L.code_box("net", lines=16)
    box_net
    return (box_net,)


@app.cell(hide_code=True)
def check_net(box_net):
    result_net = L.check("net", box_net.value)
    result_net
    return (result_net,)


@app.cell(hide_code=True)
def hints_net():
    L.hints(
        "net",
        [
            "Steps 1 and 2 of the recipe, as in the worked example, with other sizes.",
            "`__init__` needs three lines: `super().__init__()` and the two layers. `nn.Linear` takes the number of inputs first and then the number of neurons. `forward` is one line with `return`.",
            "`self.hidden = nn.Linear(2, ___)`, `self.out = nn.Linear(___, 1)`, and `return self.out(torch.relu(___))`.",
        ],
        why="The hidden layer gets the 2 inputs and has 3 neurons: `nn.Linear(2, 3)`. The output layer gets those 3 and has 1 neuron: `nn.Linear(3, 1)`. `forward` sends `x` through `self.hidden`, then ReLU, then `self.out`. That is 2 · 3 + 3 + 3 + 1 = 13 parameters.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_train():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Training it on XOR

        The class of exercise 2 is there, and the forward pass with the loss. Replace the `...` with the rest of each step of the loop: step 4 of the recipe, for every parameter of `net`. The last line is the loss of the last step, as a Python number. The check also tries another learning rate.
        """),
        L.task("Make each step set every `.grad` of the network to `None`, run the backward pass and move every parameter against its slope."),
    ])
    return


@app.cell(hide_code=True)
def exercise_train():
    box_train = L.code_box("train", lines=24)
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
            "Step 4 of the recipe, as at the end of the worked example.",
            "Two loops over `net.parameters()`: one before `loss.backward()`, to set every `.grad` to `None`, and one inside `with torch.no_grad():` for the step.",
            "`for p in net.parameters():` with `p.grad = None`; then `loss.___()`; then `with torch.no_grad():` with `for p in net.parameters():` and `p -= ___`.",
        ],
        why="Each step starts without old slopes, gets new ones for all 13 parameters from one `loss.backward()`, and moves each parameter a little against its slope. After 500 steps the network gives almost 0, 1, 1 and 0: XOR.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** `nn.Linear` is the layer of [The forward pass in NumPy, with shapes](/course/open/nn-forward-numpy?lang=en), $\mathbf{X}\mathbf{W}^\top + \mathbf{b}$, with its weights kept for you, and a class on `nn.Module` is your own `MLP` in PyTorch: layers as attributes, a `forward`, and `parameters()` for all of them. The training loop of [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en) stays the same; it only loops over `net.parameters()`. In the next lesson, *Let the optimizer take the step*, PyTorch also takes over the two loops in each step.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_net, result_train):
    L.wrap_up(choice_shape, result_net, result_train)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
