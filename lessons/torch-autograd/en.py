# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson torch-autograd. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("torch-autograd", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [PyTorch tensors](/course/open/torch-tensors?lang=en) and [The full backward pass, in the right order](/course/open/nn-micrograd-backward?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sum():
    quick_sum = L.choice(
        "quick-sum",
        question="`t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])`. What does `t.sum()` give back?",
        options={
            "number": "The Python number 10.0",
            "rows": "`tensor([3., 7.])`: one sum per row",
            "right": "`tensor(10.)`: a tensor with no axes",
        },
        correct="right",
        why={
            "number": "A summary of a tensor is a tensor again. What takes the number out? See *A tensor with no axes* in [PyTorch tensors](/course/open/torch-tensors?lang=en).",
            "rows": "Without a `dim`, `.sum()` adds up all four numbers. See [PyTorch tensors](/course/open/torch-tensors?lang=en).",
            "right": "All four numbers added up, in a tensor with no axes; `.item()` would give the Python number 10.0.",
        },
    )
    quick_sum
    return (quick_sum,)


@app.cell(hide_code=True)
def quick_check_twice():
    quick_twice = L.choice(
        "quick-twice",
        question="With your `Value` and `backward` from [The full backward pass, in the right order](/course/open/nn-micrograd-backward?lang=en): `a = Value(3)` and `b = a * a`. What is `a.grad` after `backward(b)`?",
        options={
            "one": "3",
            "right": "6",
            "square": "9",
        },
        correct="right",
        why={
            "one": "`a` is both parents of `b`. How many times does `backward_step()` add to `a.grad`? See the worked example in [The full backward pass, in the right order](/course/open/nn-micrograd-backward?lang=en).",
            "square": "9 is `b.data`, the value of `b`, not a slope. See the worked example in [The full backward pass, in the right order](/course/open/nn-micrograd-backward?lang=en).",
            "right": "`a` is both parents of `b`, each with local gradient 3, and `backward_step()` adds both: 3 + 3 = 6, the slope of $a^2$ at 3.",
        },
    )
    quick_twice
    return (quick_twice,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [The full backward pass, in the right order](/course/open/nn-micrograd-backward?lang=en) your `Value` remembered where each number came from, and your `backward` found every slope, for a graph of single numbers. In [PyTorch tensors](/course/open/torch-tensors?lang=en) a tensor holds thousands of numbers at once, but it remembers nothing yet. PyTorch's **autograd** brings the two together: tensors that remember, and a `backward()` for whole tensors at once. It does what your project does, and gives the same slopes. From here on, it gives the slopes for every network in this course, up to the language model at the end.

    **An everyday example.** A till receipt lists every item that was scanned, in order. With it, the shop can work out afterwards how much each item added to the total, without scanning anything again.
    """)
    return


@app.cell(hide_code=True)
def predict_between():
    guess_between = L.guess(
        "between",
        question="With your `Value`, every value in the graph got a `grad`, also the ones in between. In PyTorch: `w = torch.tensor(2.0, requires_grad=True)`, `z = w * 3`, `loss = z * z`, and then `loss.backward()`. What do you think `z.grad` is?",
        options={
            "right": "`None`: PyTorch keeps a slope only for `w`",
            "slope": "`tensor(12.)`: the slope of the loss with respect to `z`",
            "zero": "`tensor(0.)`",
        },
        reveal="**None.** PyTorch passes the slope back through `z` on its way to `w`, exactly as your `backward` did, but it keeps a `.grad` only for the tensors you made yourself with `requires_grad=True`: the parameters. Those are what a training step changes; the slopes in between are thrown away, to save memory. `w.grad` is 36: $\\mathcal{L} = (3w)^2 = 9w^2$, whose slope at $w = 2$ is $18 \\cdot 2 = 36$. When you read `z.grad`, PyTorch also prints a warning.",
    )
    guess_between
    return (guess_between,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Autograd** is PyTorch's automatic differentiation: what your `Value` and `backward` did, for tensors. It has three parts.

        **Recording.** A tensor made with `requires_grad=True` is one PyTorch keeps slopes for, such as a parameter. Every calculation with it gives a result that remembers how it was made, in its **`grad_fn`**: `c = a * b` has `grad_fn=<MulBackward0>`, the multiplication with its local gradients. That is what `parents` did in your `Value`: the graph is recorded while the forward pass runs.

        **The backward pass.** `loss.backward()` does what your `backward(loss)` did: it walks the recorded graph back from the loss, in the right order, starting at 1, and each step multiplies by a local gradient. It needs a loss with one number in it. The slope of every **leaf tensor** with `requires_grad=True`, a tensor you made yourself rather than the result of a calculation, lands in its `.grad`, with the same shape: a `w` with two weights gets two slopes. The tensors in between pass their slopes on, but keep none.

        **Adding up, and not recording.** As in your `Value`, `.grad` adds every new slope to the old one. So a training step sets `w.grad = None` before `loss.backward()`: PyTorch's way of zeroing the gradients, where `None` means no slope yet. And the step itself must not be recorded: it changes the parameters, and is not part of the model. Inside **`with torch.no_grad():`** PyTorch records nothing, and there `w -= lr * w.grad` changes `w` in place. Outside it, PyTorch stops with "a leaf Variable that requires grad is being used in an in-place operation". `torch.no_grad()` also works as a decorator: with `@torch.no_grad()` above a `def`, with its parentheses, nothing in that function is recorded, as you want for a function that only makes predictions.

        | In your project | In PyTorch |
        |---|---|
        | `a = Value(2)` | `a = torch.tensor(2.0, requires_grad=True)` |
        | `c.parents` | `c.grad_fn` |
        | `backward(loss)` | `loss.backward()` |
        | `a.grad` | `a.grad`, for a leaf tensor only |
        | `p.grad = 0` | `p.grad = None` |
        | `p.data = p.data - lr * p.grad` | `p -= lr * p.grad`, inside `with torch.no_grad():` |

        The neuron of exercise 3 in [The full backward pass, in the right order](/course/open/nn-micrograd-backward?lang=en), with its two weights in one tensor $\mathbf{w}$ and its inputs in one tensor $\mathbf{x}$:
        """),
        L.formula(r"[[pred:\hat{y}]] = \mathrm{ReLU}([[param:\mathbf{w}]] \cdot \mathbf{x} + [[param:b]]), \qquad [[loss:\mathcal{L}]] = ([[pred:\hat{y}]] - y)^2"),
        mo.md(r"""
        `loss.backward()` gives `w.grad` one slope per weight, in the shape of `w`, and `b.grad` one number:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial \mathbf{w}}]] = \left([[grad:\frac{\partial \mathcal{L}}{\partial w_1}]],\ [[grad:\frac{\partial \mathcal{L}}{\partial w_2}]]\right)"),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **autograd** | auto-grad | PyTorch's automatic differentiation: it records the graph and works out every slope, as your `Value` and `backward` did | |
        | `requires_grad=True` | | "keep slopes for this tensor": PyTorch records every calculation with it | `torch.tensor(2.0, requires_grad=True)` |
        | `grad_fn` | grad f n | how a result was made, with its local gradients: its link back in the graph, as `parents` was | `c.grad_fn` |
        | **leaf tensor** | | a tensor you made yourself, not the result of a calculation; with `requires_grad=True`, it gets a `.grad` | `w.is_leaf` |
        | `.backward()` | | the backward pass from a loss with one number in it: it fills the `.grad` of every leaf tensor | `loss.backward()` |
        | `.grad` | | the slope of the loss with respect to a leaf tensor, in its shape; `None` before the first backward pass | `w.grad` |
        | `-=` | minus equals | `w -= a` is short for `w = w - a`; on a tensor it changes the tensor itself, in place, instead of making a new one | `w -= lr * w.grad` |
        | `torch.no_grad()` | | a block, or a decorator, in which PyTorch records nothing: for the step, and for predictions | `with torch.no_grad():` |
        | `torch.relu` | | ReLU on every number of a tensor | `torch.relu(z)` |
        """),
        L.watch_out("`w = w - lr * w.grad` looks like the same step, but it makes a new tensor: the result of a calculation, no longer a leaf tensor. The next `loss.backward()` gives it no `.grad`, and the step after that stops, because its `.grad` is `None`. Change the tensor itself, with `-=`, inside `torch.no_grad()`."),
        mo.md(r"""
        The graph PyTorch records for $d = a \cdot b + a$, with $a = 2$ and $b = 3$, as in [Build your own autograd: a Value that remembers](/course/open/nn-micrograd-value?lang=en). Next to each arrow is its local gradient; under each value, the slope of $d$ with respect to it.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_graph():
    mo.vstack([
        C.picture({"title": "The graph PyTorch records for d = a · b + a",
                   "alt": "a = 2 and b = 3 go into c = a · b = 6, and c and a go into d = c + a = 8. Local gradients: 3 from a to c, 2 from b to c, 1 from c to d and 1 from a to d. Slopes of d: 4 for a, 2 for b, 1 for c and 1 for d."}),
        mo.md(r"""
        PyTorch records `c` with `grad_fn=<MulBackward0>` and `d` with `grad_fn=<AddBackward0>`. `d.backward()` gives the slopes 4 for $a$, 3 + 1 along its two paths, and 2 for $b$: the same as with your `Value`. Only $a$ and $b$ are leaf tensors, so only they keep their slope, in `a.grad` and `b.grad`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To get slopes with autograd, and take a step:

    1. Make the parameters with `requires_grad=True`.
    2. Forward pass: work out the loss with tensors. PyTorch records the graph.
    3. Set every `.grad` to `None`, and call `loss.backward()`.
    4. Read the slopes in `.grad`, and take the step inside `with torch.no_grad():`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    First the graph of the picture, then the linear model of [PyTorch tensors](/course/open/torch-tensors?lang=en), with one step. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_autograd():
    import torch

    # 1. the parameters of d = a * b + a
    _a = torch.tensor(2.0, requires_grad=True)
    _b = torch.tensor(3.0, requires_grad=True)
    # 2. forward: PyTorch records the graph
    _c = _a * _b
    _d = _c + _a
    print(_c, _d)
    # 3. the backward pass from d
    _d.backward()
    # 4. the slopes of the leaf tensors
    print(_a.grad, _b.grad)

    # the linear model, with the same four steps
    _X = torch.tensor([[1.0, 2.0], [3.0, 1.0], [0.0, 2.0], [2.0, 2.0]])
    _y = torch.tensor([5.0, 4.0, 3.0, 6.0])
    # 1.
    _w = torch.tensor([1.0, 1.5], requires_grad=True)
    _bias = torch.tensor(0.5, requires_grad=True)
    # 2.
    _loss = ((_X @ _w + _bias - _y) ** 2).mean()
    print(_loss)
    # 3.
    _w.grad = None
    _bias.grad = None
    _loss.backward()
    # 4.
    print(_w.grad, _bias.grad)
    with torch.no_grad():
        _w -= 0.1 * _w.grad
        _bias -= 0.1 * _bias.grad
    print(_w, _bias)
    print(((_X @ _w + _bias - _y) ** 2).mean().item())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `c` and `d` print with their `grad_fn`: `MulBackward0` and `AddBackward0`. After `d.backward()`, `a.grad` is 4 and `b.grad` is 2, as in the picture. The linear model starts with the loss 0.4375 of [PyTorch tensors](/course/open/torch-tensors?lang=en). `w.grad` holds one slope per weight: 0.75 for the first, and 0 for the second, because changing it makes some predictions better and others worse by just as much. The step moves `w` to (0.925, 1.5) and the bias term to 0.475, and the loss goes down to about 0.4009. After the step, `w` still says `requires_grad=True`: it is the same leaf tensor, ready for the next step.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_neuron():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The neuron again

        The neuron of exercise 3 in [The full backward pass, in the right order](/course/open/nn-micrograd-backward?lang=en): inputs 2 and 1, label 3, weights 0.5 and −1, and bias term 1. Now its two weights are one tensor `w`, and its inputs one tensor `x`. Replace the `...` with the forward pass, ending in `loss`, and then `loss.backward()`. The last line shows the slopes. The check also tries other weights and another bias term, one of them with `z` below 0.
        """),
        L.task("Let autograd give the slopes of the loss with respect to `w` and `b`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_neuron():
    box_neuron = L.code_box("neuron", lines=11)
    box_neuron
    return (box_neuron,)


@app.cell(hide_code=True)
def check_neuron(box_neuron):
    result_neuron = L.check("neuron", box_neuron.value)
    result_neuron
    return (result_neuron,)


@app.cell(hide_code=True)
def hints_neuron():
    L.hints(
        "neuron",
        [
            "Steps 2 and 3 of the recipe. The formula is in *The idea*.",
            "`w @ x` is the dot product of the weights and the inputs, and ReLU on a tensor is `torch.relu`. The `.grad`s are still `None` here, so before this one backward pass there is nothing to set to `None`.",
            "`z = w @ x + ___`, `y_pred = torch.relu(z)`, `loss = (y_pred - y) ** 2` and `loss.___()`.",
        ],
        why="$z = 0.5 \\cdot 2 - 1 \\cdot 1 + 1 = 1$, so $\\hat{y} = 1$ and $\\hat{y} - y = -2$. The slope of the loss with respect to $z$ is $2 \\cdot (-2) = -4$, and for each weight that times its input: $-4 \\cdot 2 = -8$ and $-4 \\cdot 1 = -4$. For $b$ it is $-4$ itself.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.md(r"""
    ### Exercise 2. A step that PyTorch refuses
    """)
    return


@app.cell(hide_code=True)
def choice_step_cell():
    choice_step = L.choice(
        "step",
        question="A training step `w -= lr * w.grad` stops PyTorch with \"a leaf Variable that requires grad is being used in an in-place operation\". What fixes it?",
        options={
            "rebind": "`w = w - lr * w.grad` instead of `-=`",
            "backward": "Calling `loss.backward()` once more before the step",
            "right": "The same line, inside `with torch.no_grad():`",
        },
        correct="right",
        why={
            "rebind": "That runs once, but then `w` is the result of a calculation, not a leaf tensor: the next `loss.backward()` gives it no `.grad`. See the watch-out in *The idea*.",
            "backward": "A second backward pass over the same graph stops PyTorch too, and the step would still be recorded. What leaves the step out of the graph? See *The idea*.",
            "right": "Inside `torch.no_grad()` PyTorch records nothing, so it lets you change the parameter in place. It stays the same leaf tensor, with `requires_grad=True`.",
        },
    )
    choice_step
    return (choice_step,)


@app.cell(hide_code=True)
def text_exercise_train():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A training loop with autograd

        The linear model of the worked example, trained from `w` and `b` at 0, in 100 steps with the learning rate `lr`. The forward pass is there. Replace the `...` with the rest of each step: steps 3 and 4 of the recipe, for `w` and `b`. The last line is the loss of the last step, as a Python number. The check also tries another learning rate.
        """),
        L.task("Make each step of the loop zero the gradients, run the backward pass and take a step against the slopes."),
    ])
    return


@app.cell(hide_code=True)
def exercise_train():
    box_train = L.code_box("train", lines=17)
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
            "Steps 3 and 4 of the recipe, for both `w` and `b`, as at the end of the worked example.",
            "Inside the loop, the lines of the `with` block go one level deeper. Use `-=`, so that `w` stays the same leaf tensor.",
            "`w.grad = None`, `b.grad = None` and `loss.___()`; then `with torch.no_grad():` with `w -= lr * w.grad` and `b -= ___`.",
        ],
        why="Each step starts without old slopes, gets new ones from `loss.backward()`, and moves `w` and `b` a little against them. The loss goes from 21.5, with everything at 0, to about 0.33.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Autograd is your `Value` and your `backward` from [The full backward pass, in the right order](/course/open/nn-micrograd-backward?lang=en), built into the tensors of [PyTorch tensors](/course/open/torch-tensors?lang=en): `requires_grad` records, `backward()` walks back, `.grad` adds up, and the step happens inside `torch.no_grad()`, which is also a decorator, as in [Decorators](/course/open/py-decorators?lang=en). In the next lesson, *Layers as modules: nn.Linear*, PyTorch also keeps the weights and the forward pass of a layer together, as your `Layer` and `MLP` did.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_neuron, choice_step, result_train):
    L.wrap_up(result_neuron, choice_step, result_train)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
