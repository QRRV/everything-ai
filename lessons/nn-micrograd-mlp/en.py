# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-micrograd-mlp. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("nn-micrograd-mlp", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [The full backward pass, in the right order](/course/open/nn-micrograd-backward?lang=en) and [Building on another class](/course/open/py-inheritance?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_backward():
    quick_backward = L.choice(
        "quick-backward",
        question="With your `Value` and your `backward`: `a = Value(3)`, `b = Value(4)`, `c = a * b + a`, then `backward(c)`. What is `a.grad`?",
        options={
            "right": "5",
            "one": "4",
            "two": "1",
        },
        correct="right",
        why={
            "one": "That is what comes back through `a * b` alone. `a` also goes straight into the sum. See the watch-out in [Build your own autograd: a Value that remembers](/course/open/nn-micrograd-value?lang=en).",
            "two": "That is what comes back along the arrow straight into the sum. `a` also goes into `a * b`. See the watch-out in [Build your own autograd: a Value that remembers](/course/open/nn-micrograd-value?lang=en).",
            "right": "4 back through `a * b`, the number of `b`, and 1 along the arrow straight into the sum: together 5, the slope of $ab + a$ for $a$.",
        },
    )
    quick_backward
    return (quick_backward,)


@app.cell(hide_code=True)
def quick_check_inherit():
    quick_inherit = L.choice(
        "quick-inherit",
        question="`class Pet:` has a method `name_tag(self)`, and `class Dog(Pet):` does not. What does `Dog().name_tag()` do?",
        options={
            "error": "Python stops: `Dog` has no `name_tag`",
            "right": "It runs the `name_tag` of `Pet`",
            "empty": "It gives back nothing, because `Dog` has no method of its own",
        },
        correct="right",
        why={
            "error": "Python looks in the object's own class first, and then where? See *The idea* in [Building on another class](/course/open/py-inheritance?lang=en).",
            "empty": "A subclass gets all the methods of its parent class. See *The idea* in [Building on another class](/course/open/py-inheritance?lang=en).",
            "right": "`Dog` is built on `Pet`, so it gets its methods: Python finds `name_tag` in `Pet`.",
        },
    )
    quick_inherit
    return (quick_inherit,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        In [The full backward pass, in the right order](/course/open/nn-micrograd-backward?lang=en) the demo built a network with a hidden layer out of loose `Value`s: 17 parameters, each with its own place in a list. A network for real data has thousands, in layers of different sizes. Writing that out by hand is no longer doable. A network is built from a few kinds of parts that repeat: neurons, layers of neurons, and layers one after another. Today you write those parts as classes in your project. That is the sixth step of your own neural network.
        """),
        draw.project_steps(["Linear regression on many features", "Training logistic regression", "A linear classifier for handwritten digits",
                            "Your own autograd: a Value that remembers", "The full backward pass", "Neurons, layers and an MLP",
                            "Training your MLP", "An MLP in PyTorch that reads handwritten digits", "Backprop through whole layers by hand",
                            "A reusable training loop", "Make a deep network train"],
                           5, here_label="← today", title="Where your project stands: your own neural network"),
        L.course_versions(),
        mo.md(r"""
        **An everyday example.** A house is built from bricks, the bricks make walls, and the walls make rooms. Once you know how to lay one brick, you can build a house of any size.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_count():
    guess_count = L.guess(
        "count",
        question="A neuron with 3 inputs has 3 weights and a bias term. A layer of 4 such neurons gets the same 3 inputs. How many parameters do you think the layer has?",
        options={
            "weights": "12",
            "right": "16",
            "neurons": "7",
        },
        reveal="**16.** Each of the 4 neurons has its own 3 weights and its own bias term: 4 × (3 + 1) = 16. In a layer, every neuron sees every input, so a layer of $m$ neurons with $n$ inputs has $m \\cdot (n + 1)$ parameters: the $m \\times n$ weights of its weight matrix, and $m$ bias terms.",
    )
    guess_count
    return (guess_count,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Three classes, each built on the one before:

        - A `Neuron` holds a list of weights `w`, one `Value` per input, and a bias term `b`. Called on the inputs `x`, a list of `Value`s, it works out $z = b + w_1 x_1 + w_2 x_2 + \dots$, and gives back $\mathrm{ReLU}(z)$, or $z$ itself in the output layer.
        - A `Layer` holds a list of neurons that all get the same inputs. Called on `x`, it gives back the list of their outputs: the inputs of the next layer.
        - An `MLP` holds a list of layers. Called on `x`, it passes the outputs of each layer on to the next, and gives back those of the last.

        Each of them has a method `parameters()` that gives back all its `Value`s that training may change: a neuron its weights and bias term, a layer those of all its neurons, an MLP those of all its layers. They all build on one small class, `Module`, whose `parameters()` gives back an empty list. The weights start at random numbers between −1 and 1, so that the neurons of a layer start out different; the bias terms start at 0.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | `Neuron`, `Layer`, `MLP` | | classes for one neuron, a layer of neurons, and layers one after another | `MLP(2, [4, 4, 1])` |
        | `parameters()` | | a method that gives back every `Value` training may change | `m.parameters()` |
        | `Module` | | the class all three build on; its `parameters()` gives back `[]` | `class Neuron(Module):` |
        | `random.uniform(a, b)` | | a random number with decimals between `a` and `b` | `random.uniform(-1, 1)` |
        """),
        L.watch_out("Starting every weight at 0 would make every neuron of a layer the same: the same output, the same slope, the same step, for ever. Random starting weights make them different, so each can learn something of its own."),
        L.watch_out("`Module` is the name of a class here: the class the others build on. A Python module is still a file of code, such as `neuron.py`; that file holds the class `Module`."),
        mo.md(r"""
        `MLP(2, [4, 4, 1])`: 2 inputs, two hidden layers of 4 neurons, and one output neuron. Every line is a weight, and every circle after the inputs also has its own bias term.
        """),
        draw.network_diagram([2, 4, 4, 1], ["2 inputs", "4 neurons, ReLU", "4 neurons, ReLU", "1 output"],
                             title="MLP(2, [4, 4, 1])",
                             alt="Four columns of circles: 2 inputs, two hidden layers of 4 neurons and 1 output, with a line from every circle to every circle of the next column."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To build a network from your `Value`:

    1. A neuron: `z` starts at `b`; add `w[i] * x[i]` for every input; give back `z.relu()`, or `z` in the output layer.
    2. A layer: call each of its neurons on the same `x`, and give back the list of their outputs.
    3. An MLP: pass `x` through each layer in turn; the output of one is the input of the next.
    4. `parameters()`: put the parameters of the parts together in one list, `+` between lists.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Counting the parameters of `MLP(2, [4, 4, 1])`, layer by layer, the way `parameters()` will. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_count():
    _sizes = [2, 4, 4, 1]
    _n_inputs = _sizes[0]
    _total = 0
    for _i in range(1, len(_sizes)):
        _n_neurons = _sizes[_i]
        # 1. one neuron: a weight per input, and a bias term
        _per_neuron = _n_inputs + 1
        # 2. a layer: that, for each of its neurons
        _per_layer = _n_neurons * _per_neuron
        print(f"layer {_i}: {_n_neurons} × {_per_neuron} = {_per_layer}")
        # 3. the MLP: the outputs of this layer are the inputs of the next
        _n_inputs = _n_neurons
        # 4. parameters(): those of all layers together
        _total = _total + _per_layer
    print("in total", _total)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    12 + 20 + 5 = 37 parameters. Step 3 is the line `_n_inputs = _n_neurons`: a layer's inputs are the outputs of the layer before it, so the second hidden layer has 4 inputs, not 2. Your `MLP` will make these 37 `Value`s, and `backward` gives each one its slope.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_count():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. How many parameters?
    """)
    return


@app.cell(hide_code=True)
def choice_size_cell():
    choice_size = L.choice(
        "size",
        question="How many parameters does `MLP(3, [4, 4, 1])` have: 3 inputs, two hidden layers of 4 neurons, and one output neuron?",
        options={
            "weights": "32",
            "neurons": "9",
            "right": "41",
        },
        correct="right",
        why={
            "weights": "Those are the weights: 12, 16 and 4. Each neuron also has a bias term. See the worked example.",
            "neurons": "That is the number of neurons. Each has a weight per input, and a bias term. See the worked example.",
            "right": "4 × (3 + 1) = 16, then 4 × (4 + 1) = 20, then 1 × (4 + 1) = 5: 41.",
        },
    )
    choice_size
    return (choice_size,)


@app.cell(hide_code=True)
def text_exercise_neuron():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Your Neuron

        Finish the class `Neuron`. `Module` and `__init__` are done: `self.w` is the list of weights, `self.b` the bias term, and `self.relu` is `True` or `False`. `relu=True` in the `def` line gives `relu` a value for when a call leaves it out: `Neuron(3)` gets ReLU, `Neuron(3, relu=False)` does not. Replace the two `...`: `__call__` works out step 1 of the recipe on the inputs `x`, a list of your `Value`s; `parameters()` gives back the weights and the bias term in one list. The check makes neurons, sets their weights, calls them, and runs your `backward` on what comes out. When it is right, the check saves the box in your project, as `work/projects/nn/neuron.py`.
        """),
        L.task("Make a `Neuron` work out its output, and give back its parameters."),
    ])
    return


@app.cell(hide_code=True)
def exercise_neuron():
    box_neuron = L.code_box("neuron", lines=27)
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
            "Step 1 of the recipe: a loop over the inputs, as in the forward pass of your lessons before, but with the neuron's own `self.w` and `self.b`.",
            "Start `z` at the neuron's own bias term, and in a loop over `range(len(self.w))` add one weight times its input in each round. `self.relu` says whether ReLU comes after. `parameters()` puts the list `self.w` and a list with `self.b` together.",
            "`z = ___`, then `for i in range(len(self.w)):` with `z = z + ___ * ___`; then `if self.relu:` with `return z.relu()`, and `return z`. In `parameters()`: `return self.w + [___]`.",
        ],
        why="With the weights 0.5 and −1, the bias term 1 and the inputs 2 and 1: $z = 1 + 0.5 \\cdot 2 - 1 \\cdot 1 = 1$, above 0, so ReLU passes 1 on. After `backward`, the weights have the slopes 2 and 1, their inputs, and the bias term 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mlp():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Layers and an MLP

        The box imports your `Module` and your `Neuron` from your project. `Layer` makes its neurons and gives back their parameters; `MLP` makes its layers, the last one without ReLU. Replace the three `...`: `Layer.__call__` gives back the list of the outputs of its neurons, `MLP.__call__` passes `x` through every layer, and `MLP.parameters()` puts the parameters of all layers together, as `Layer.parameters()` does for its neurons. The check builds MLPs of different sizes, calls them and runs your `backward`. When it is right, the check saves the box in your project, as `work/projects/nn/mlp.py`.
        """),
        L.task("Make a `Layer` and an `MLP` work out their outputs, and let an `MLP` give back all its parameters."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mlp():
    box_mlp = L.code_box("mlp", lines=34)
    box_mlp
    return (box_mlp,)


@app.cell(hide_code=True)
def check_mlp(box_mlp, result_neuron):
    result_neuron  # check again when exercise 2 saves your module
    result_mlp = L.check("mlp", box_mlp.value)
    result_mlp
    return (result_mlp,)


@app.cell(hide_code=True)
def hints_mlp():
    L.hints(
        "mlp",
        [
            "Steps 2, 3 and 4 of the recipe. `Layer.parameters()` shows how to put lists together.",
            "Call each neuron like a function, on the same list `x`, and collect what comes out in a list. In `MLP.__call__`, the name `x` gets a new value in each round of a loop over `self.layers`: what that layer makes of it.",
            "`Layer.__call__`: `outputs = []`, then for each `n` in `self.neurons`: `outputs.append(___)`, and `return outputs`. `MLP.__call__`: `for layer in self.layers:` with `x = ___`, and `return x`.",
        ],
        why="With every parameter at 0.5, `MLP(2, [2, 1])` on the inputs 1 and 2 gives each hidden neuron $0.5 + 0.5 + 1 = 2$, which ReLU passes on, and the output neuron $0.5 + 0.5 \\cdot 2 + 0.5 \\cdot 2 = 2.5$. It has $2 \\cdot 3 + 1 \\cdot 3 = 9$ parameters.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell below builds `MLP(2, [4, 4, 1])` with your own classes, from the same random starting weights every time, and calls it on the four corners of XOR. Then it builds the loss over the four corners and calls your `backward` once. The cell uses your own Python modules, so it works once exercise 3 is done.
    """)
    return


@app.cell(hide_code=True)
def demo_mlp(result_mlp):
    result_mlp  # run again when exercise 3 saves your module
    _out = C.demo()
    if _out is None:
        _view = mo.md("*Finish exercise 3 first: then your `MLP` is in your project, and this cell uses it.*")
    elif not _out:
        _view = mo.md("*Your `neuron.py` or `mlp.py` in `work/projects/nn/` no longer works the way its check expects. Did you change one of them? Delete it, and check its exercise again: its check then saves it again.*")
    else:
        _n, _outputs, _loss, _with_slope, _own = _out
        _view = mo.md(f"""Your MLP has <span class="ea-num">{_n}</span> parameters. Untrained, it gives <span class="c-pred ea-num">{", ".join(L.number(_o, 2) for _o in _outputs)}</span> for (0, 0), (0, 1), (1, 0) and (1, 1), where XOR wants 0, 1, 1 and 0: the loss is <span class="c-loss ea-num">{L.number(_loss, 2)}</span>. One call of {"your `backward`" if _own else "the course's `backward`"} gave <span class="ea-num">{_with_slope}</span> of the {_n} parameters a slope other than 0. The others are weights that get nothing back: at each corner, either their input is 0, or the ReLU of their neuron gives 0 there. In the next lesson you take steps against these slopes, again and again: training.""")
    _view
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [A hidden layer and ReLU](/course/open/nn-hidden-layer?lang=en) worked out a network neuron by neuron; now classes that build on each other, as in [Building on another class](/course/open/py-inheritance?lang=en), do that for a network of any size, on top of the `Value` and `backward` of [The full backward pass, in the right order](/course/open/nn-micrograd-backward?lang=en). Your project `nn` now has `neuron.py` and `mlp.py`. In the next lesson, *Training your MLP: the training loop*, you train such an MLP: forward, loss, backward, step, and again.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_size, result_neuron, result_mlp):
    L.wrap_up(choice_size, result_neuron, result_mlp)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
