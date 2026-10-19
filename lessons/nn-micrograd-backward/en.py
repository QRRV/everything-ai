# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-micrograd-backward. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("nn-micrograd-backward", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Build your own autograd: a Value that remembers](/course/open/nn-micrograd-value?lang=en) and [A function that calls itself](/course/open/py-recursion?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_parents():
    quick_parents = L.choice(
        "quick-parents",
        question="With your `Value`: `a = Value(2)`, `b = Value(3)` and `c = a * b`. What is in `c.parents`?",
        options={
            "own": "`[(a, 2), (b, 3)]`",
            "right": "`[(a, 3), (b, 2)]`",
            "plain": "`[a, b]`",
        },
        correct="right",
        why={
            "own": "Each parent is paired with its local gradient. How fast does $a \\cdot b$ change when $a$ changes? See *The idea* in [Build your own autograd: a Value that remembers](/course/open/nn-micrograd-value?lang=en).",
            "plain": "`parents` holds pairs: the value, and its local gradient. See *The idea* in [Build your own autograd: a Value that remembers](/course/open/nn-micrograd-value?lang=en).",
            "right": "`a` with the number of `b`, 3, and `b` with the number of `a`, 2: the local gradients of a product.",
        },
    )
    quick_parents
    return (quick_parents,)


@app.cell(hide_code=True)
def quick_check_base():
    quick_base = L.choice(
        "quick-base",
        question="A recursive function calls itself, but has no base case. What happens when you call it?",
        options={
            "zero": "It gives back 0",
            "forever": "It runs forever, and you have to stop it yourself",
            "right": "Python stops it with an error, after about a thousand calls",
        },
        correct="right",
        why={
            "zero": "Without a base case it never reaches a `return` with a value. What does Python do with calls that never end? See [A function that calls itself](/course/open/py-recursion?lang=en).",
            "forever": "Python keeps count of the calls that are still open, and gives up at a limit. See *The idea* in [A function that calls itself](/course/open/py-recursion?lang=en).",
            "right": "Each call waits for the next, and Python allows about a thousand open calls; after that, Python stops with an error.",
        },
    )
    quick_base
    return (quick_base,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        In [Build your own autograd: a Value that remembers](/course/open/nn-micrograd-value?lang=en) you called `backward_step()` on every value yourself, in an order you worked out, and the demo used an order the course wrote. A network with a hidden layer has hundreds of values, and every new model a different graph. But the order is already in the graph: each `Value` knows where it came from. Today your project gets a `backward()` that finds the order by itself, for any graph. That is the fifth step of your own neural network.
        """),
        draw.project_steps(["Linear regression on many features", "Training logistic regression", "A linear classifier for handwritten digits",
                            "Your own autograd: a Value that remembers", "The full backward pass", "Neurons, layers and an MLP",
                            "Training your MLP", "An MLP in PyTorch that reads handwritten digits", "Backprop through whole layers by hand",
                            "A reusable training loop", "Make a deep network train"],
                           4, here_label="← today", title="Where your project stands: your own neural network"),
        L.course_versions(),
        mo.md(r"""
        **An everyday example.** Getting dressed: socks before shoes, a shirt before a sweater. Any order that keeps those rules works. To get undressed, you take the same order backwards: shoes before socks.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_order():
    guess_order = L.guess(
        "order",
        question="$c = a \\cdot b$ and $d = c + a$. To make a list, you start at `d`: first you list everything `d` was made from, in the same way, and only then `d` itself. What do you think the list is?",
        options={
            "start": "`[d, c, a, b]`",
            "twice": "`[a, b, c, a, d]`",
            "right": "`[a, b, c, d]`",
        },
        reveal="**`[a, b, c, d]`.** Every value comes after the values it was made from, and each value is in the list once: `a` too, although both `c` and `d` were made from it. In *The idea* you follow how the list is built, one step at a time.",
    )
    guess_order
    return (guess_order,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **topological order** of a graph is a list of its values in which every value comes after all the values it was made from. A recursive function finds one, starting at the loss:

        1. If the value is already in the list, stop: that is the base case.
        2. Otherwise, first do the same for each of its parents.
        3. Then add the value itself to the list.

        A value with no parents just goes in. To remember which values were already visited, the function keeps a `set`, `seen`: `seen.add(v)` puts `v` in, and `v in seen` asks whether it is there. Every call gets the same `seen` and the same `order`: each call has its own names, as in [A function that calls itself](/course/open/py-recursion?lang=en), but they all point to one set and one list, so what one call adds, the others see.

        The backward pass takes this order backwards: the loss first, and every value only after all the values it went into. So by the time a value passes its `grad` on with `backward_step()`, it has all of it.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **topological order** | | a list of the values of a graph in which every value comes after the values it was made from | `order` |
        | `set()` | | a new, empty set | `build_order(loss, set(), order)` |
        | `seen.add(v)` | | puts `v` in the set `seen`; a value that is already there stays there once | `seen.add(v)` |
        | `v in seen` | | `True` when `v` is in the set `seen` | `if v in seen:` |
        | `reversed(order)` | | goes through a list from the end to the start, without changing it | `for v in reversed(order):` |
        """),
        L.watch_out("Every value must be in the list once. A value that goes into two nodes would be visited twice; listed twice, its `backward_step()` runs twice and passes its slope on twice. `seen` stops that, and in a large graph it also saves the time of visiting the same values again and again."),
        mo.md(r"""
        Move the slider: it follows `build_order` from `d` in $d = a \cdot b + a$. The title of the picture names every value `build_order` has been called on so far, and a value gets its place in the list once all its parents have one.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_step_cell():
    _say = [
        "Start at `d`: `build_order(d)`. It was made from `c` and `a`, so first `build_order(c)`; `c` was made from `a` and `b`, so first `build_order(a)`. The list is still empty.",
        "`a` was made from nothing: it goes in, at place 1. Back in `c`, the next parent is `b`: `build_order(b)`.",
        "`b` was made from nothing either: place 2. Now both parents of `c` are in the list.",
        "So `c` goes in, at place 3. Back in `d`, its other parent is `a`: `build_order(a)` once more. But `a` is already in the list.",
        "Both parents of `d` are in the list, so `d` goes in, at place 4. Backwards, `d`, `c`, `b`, `a` is the order of the backward pass.",
    ]

    def _frame(step):
        step = int(step)
        return (C.picture(step, {"place": "place", "title": f"build_order called on {', '.join(C.CALLS[:C.CALLED[step]])}",
                                 "alt": f"The Value graph of d = a · b + a: a and b go into c, c and a go into d. After step {step} of 4 the list is [{', '.join(C.ORDER[:step])}]."}).text,
                _say[step])

    explore_step = L.explore("step", "step", _frame, start=0, low=0, high=4, step=1, role="", task="How many times is `build_order` called, and which call stops at once, at the base case?")
    explore_step
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    `build_order` is called five times: for `d`, `c`, `a`, `b`, and `a` once more. That last call stops at once: `a` is already in `seen`. Without `seen`, `a` would go into the list a second time: `[a, b, c, a, d]`.

    ### Recipe

    To do the whole backward pass from a loss:

    1. Make an empty list `order` and an empty set `seen`.
    2. From the loss, recursively: skip a value that is in `seen`; otherwise add it to `seen`, do its parents first, and then append it to `order`.
    3. Set the `grad` of the loss to 1.
    4. Go through `order` backwards, and call `backward_step()` on each value.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The whole recipe on a small family of values that pass their `grad` on as it is, without local gradients, so you can see the list being built and the `grad` going back. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_order():
    class _Node:
        def __init__(self, name, parents=()):
            self.name = name
            self.parents = parents
            self.grad = 0

        def backward_step(self):
            for _parent in self.parents:
                _parent.grad = _parent.grad + self.grad

    def _visit(v, seen, order):
        if v in seen:                 # 2. already there: skip it
            return
        seen.add(v)
        for _parent in v.parents:     # 2. its parents first
            _visit(_parent, seen, order)
        order.append(v)               # 2. then the value itself

    _a = _Node("a")
    _b = _Node("b")
    _c = _Node("c", [_a, _b])
    _d = _Node("d", [_c, _a])
    # 1. an empty list and an empty set
    _order = []
    _visit(_d, set(), _order)
    _names = []
    for _v in _order:
        _names.append(_v.name)
    print(_names)
    # 3. the grad of the loss, d, is 1
    _d.grad = 1
    # 4. backwards: each value passes its grad on
    for _v in reversed(_order):
        _v.backward_step()
        print(_v.name, _v.grad)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The list is `['a', 'b', 'c', 'd']`, as at step 4 above: every value after the ones it was made from. Backwards, `d` passes its 1 to `c` and `a`, and only then `c` passes its 1 to `a` and `b`: `a` gets 1 along each way back from `d`, 2 in all. These values pass their `grad` on as it is; your `Value` multiplies it by the local gradient, and its parents are pairs, `(parent, local)`. So in your own function the loop takes them apart, as `backward_step()` does.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_order():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which list is a topological order?
    """)
    return


@app.cell(hide_code=True)
def choice_topo_cell():
    choice_topo = L.choice(
        "topo",
        question="The student of [Backpropagation by hand on a tiny graph](/course/open/nn-backward-by-hand?lang=en): $m = w \\cdot x$, $\\hat{y} = m + b$, $r = 7 - \\hat{y}$ and $\\mathcal{L} = r^2$. Which list is a topological order of $w$, $x$, $b$, $m$, $\\hat{y}$, $r$ and $\\mathcal{L}$?",
        options={
            "right": "$b$, $w$, $x$, $m$, $\\hat{y}$, $r$, $\\mathcal{L}$",
            "loss": "$\\mathcal{L}$, $r$, $\\hat{y}$, $m$, $b$, $w$, $x$",
            "early": "$w$, $x$, $\\hat{y}$, $m$, $b$, $r$, $\\mathcal{L}$",
        },
        correct="right",
        why={
            "loss": "That is the order of the backward pass: the order backwards. In a topological order, does $\\mathcal{L}$ come before or after the values it was made from? See *The idea*.",
            "early": "$\\hat{y}$ is made from $m$ and $b$. Are both before it in this list? See *The idea*.",
            "right": "Every value comes after the ones it was made from: $m$ after $w$ and $x$, $\\hat{y}$ after $m$ and $b$. That $b$ comes first does not matter; more than one order is right.",
        },
    )
    choice_topo
    return (choice_topo,)


@app.cell(hide_code=True)
def text_exercise_backward():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Your backward()

        Finish `build_order` and `backward`. In `build_order`, replace the `...` with step 2 of the recipe for the parents of `v`: they are pairs, `(parent, local)`, as in `backward_step()`. In `backward`, replace the `...` with step 4. The check makes graphs with your `Value` from your project, calls your `backward` on them, and looks at the slopes, and at the order your `build_order` makes. When it is right, the check saves both functions in your project, as `work/projects/nn/backward.py`.
        """),
        L.task("Make `build_order` put the parents first, and `backward` call `backward_step()` in the reverse order."),
    ])
    return


@app.cell(hide_code=True)
def exercise_backward():
    box_backward = L.code_box("backward", lines=16)
    box_backward
    return (box_backward,)


@app.cell(hide_code=True)
def check_backward(box_backward):
    result_backward = L.check("backward", box_backward.value)
    result_backward
    return (result_backward,)


@app.cell(hide_code=True)
def hints_backward():
    L.hints(
        "backward",
        [
            "Step 2 of the recipe, as `_visit` in the worked example, but with pairs; then step 4.",
            "The loop over `v.parents` gives two names, as in `backward_step()`, and calls `build_order` on the first one. Step 4 goes through `reversed(order)`.",
            "`for parent, local in v.parents:` with `build_order(___, seen, order)`; and `for v in reversed(___):` with `v.___()`.",
        ],
        why="For $d = a \\cdot b + a$, `build_order` makes `[a, b, c, d]`, and backwards `d.backward_step()` gives `c` and `a` each 1, then `c.backward_step()` gives `a` 3 more and `b` 2. So `a.grad` is 4 and `b.grad` 2, the slopes of $ab + a$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_neuron():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A neuron with two inputs

        A neuron with ReLU and two inputs, for one example: $z = w_1 x_1 + w_2 x_2 + b$, $\hat{y} = \mathrm{ReLU}(z)$, and the loss $(\hat{y} - y)^2$. The box imports your `Value` and your `backward` from your project. Replace the `...` with the forward pass, with `+`, `*`, `relu()` and `Value(-1)` for the minus, and end it in `loss`. Then one call, `backward(loss)`, does the rest. The last line is `[w1.grad, w2.grad, b.grad]`, so the check sees it. The check also tries other numbers.
        """),
        L.task("Build the loss of the neuron with your Value, and get all three slopes with one call to your backward."),
    ])
    return


@app.cell(hide_code=True)
def exercise_neuron():
    box_neuron = L.code_box("neuron", lines=15)
    box_neuron
    return (box_neuron,)


@app.cell(hide_code=True)
def check_neuron(box_neuron, result_backward):
    result_backward  # check again when exercise 2 saves your module
    result_neuron = L.check("neuron", box_neuron.value)
    result_neuron
    return (result_neuron,)


@app.cell(hide_code=True)
def hints_neuron():
    L.hints(
        "neuron",
        [
            "Write the forward pass one operation at a time, as in exercise 3 of the lesson before; you do not need a name for every value any more.",
            "$\\hat{y} - y$ is `y_pred + y * Value(-1)`, and the square is that value times itself.",
            "`z = w1 * x1 + w2 * x2 + b`, `y_pred = z.relu()`, `r = y_pred + y * Value(-1)`, `loss = ___ * ___`, then `backward(___)`.",
        ],
        why="With $x = (2, 1)$, $y = 3$, $w = (0.5, -1)$ and $b = 1$: $z = 1 - 1 + 1 = 1$, so $\\hat{y} = 1$ and $\\hat{y} - y = -2$. Back: $2 \\cdot (-2) = -4$ for $\\hat{y}$, and $z$ is above 0, so $-4$ for $z$; then $-4 \\cdot 2 = -8$ for $w_1$, $-4 \\cdot 1 = -4$ for $w_2$, and $-4$ for $b$.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell below trains a network with a hidden layer of 4 neurons with ReLU on XOR, the task one neuron could not learn in [What one neuron cannot learn](/course/open/nn-xor?lang=en). Its 17 weights and bias terms are `Value`s of yours; each step builds the loss over the four corners and calls your own `backward`, and then moves every parameter against its slope, 500 times. The cell uses your own modules, so it works once exercise 2 is done.
    """)
    return


@app.cell(hide_code=True)
def demo_xor(result_backward):
    result_backward  # run again when exercise 2 saves your module
    _fit = C.trained()
    if _fit is None:
        _out = mo.md("*Finish exercise 2 first: then your `backward` is in your project, and this cell uses it.*")
    elif not _fit:
        _out = mo.md("*Your `backward` in `work/projects/nn/backward.py` no longer gives the right slopes, or stops with an error. Did you change that file? Delete it, and check exercise 2 again: its check then saves it again.*")
    else:
        _before, _after, _outputs, _size, _own = _fit
        _out = mo.md(f"""The loss over the four corners went from <span class="c-loss ea-num">{L.number(_before, 2)}</span> to <span class="c-loss ea-num">{L.number(_after, 4)}</span>. For (0, 0), (0, 1), (1, 0) and (1, 1) the network now gives <span class="c-pred ea-num">{", ".join(L.number(_o, 2) for _o in _outputs)}</span>: XOR. Every slope came from {"your own `Value`" if _own else "the course's `Value`"}, in an order your own `backward` found, through a graph of {_size} values per step.""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Build your own autograd: a Value that remembers](/course/open/nn-micrograd-value?lang=en) each `Value` learned to pass its slope back; with the recursion of [A function that calls itself](/course/open/py-recursion?lang=en) and a set from [Tuples, sets and sorting](/course/open/py-tuples-sets?lang=en), your `backward` now finds the order of those steps itself. Your project `nn` now has `backward.py`. In the next lesson, *Neurons, layers and an MLP on top of your Value*, you build neurons and layers as classes, so that a network of any size is a few lines.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_topo, result_backward, result_neuron):
    L.wrap_up(choice_topo, result_backward, result_neuron)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
