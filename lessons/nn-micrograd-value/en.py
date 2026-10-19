# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-micrograd-value. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("nn-micrograd-value", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Backpropagation by hand on a tiny graph](/course/open/nn-backward-by-hand?lang=en) and [Special methods: making + and print work on your objects](/course/open/py-special-methods?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_local():
    quick_local = L.choice(
        "quick-local",
        question="In a backward pass, $m = w \\cdot x$ with $x = 3$, and $\\frac{\\partial \\mathcal{L}}{\\partial m} = 2$. What is $\\frac{\\partial \\mathcal{L}}{\\partial w}$?",
        options={
            "sum": "5",
            "same": "2",
            "right": "6",
        },
        correct="right",
        why={
            "sum": "The backward pass does not add the local gradient; what does it do with it? See the recipe in [Backpropagation by hand on a tiny graph](/course/open/nn-backward-by-hand?lang=en).",
            "same": "That is the slope of $m$. What is the local gradient of $m = w \\cdot x$ for $w$? See *The idea* in [Backpropagation by hand on a tiny graph](/course/open/nn-backward-by-hand?lang=en).",
            "right": "The slope after it, 2, times the local gradient $\\frac{\\partial m}{\\partial w} = x = 3$: 6.",
        },
    )
    quick_local
    return (quick_local,)


@app.cell(hide_code=True)
def quick_check_add():
    quick_add = L.choice(
        "quick-add",
        question="`a` and `b` are objects of your own class. Which method does Python run for `a + b`?",
        options={
            "right": "`a.__add__(b)`",
            "other": "`b.__add__(a)`",
            "str": "`a.__str__()`",
        },
        correct="right",
        why={
            "other": "The object on the left of `+` is `self`. Which one is that here? See [Special methods: making + and print work on your objects](/course/open/py-special-methods?lang=en).",
            "str": "`__str__` runs for `print` and f-strings. Which method runs for `+`? See [Special methods: making + and print work on your objects](/course/open/py-special-methods?lang=en).",
            "right": "`+` runs `__add__` of the object on its left, with the one on its right as `other`.",
        },
    )
    quick_add
    return (quick_add,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        In [Backpropagation by hand on a tiny graph](/course/open/nn-backward-by-hand?lang=en) you wrote out every value and every slope yourself. For a network with thousands of weights, the code has to keep that record by itself: which value was made from which, and with what local gradient. That is **autograd**, and every library that trains networks has one. Today you build its core in your project, the way Andrej Karpathy's small *micrograd* does: a `Value` that remembers where it came from. That is the fourth step of your own neural network.
        """),
        draw.project_steps(["Linear regression on many features", "Training logistic regression", "A linear classifier for handwritten digits",
                            "Your own autograd: a Value that remembers", "The full backward pass", "Neurons, layers and an MLP",
                            "Training your MLP", "An MLP in PyTorch that reads handwritten digits", "Backprop through whole layers by hand",
                            "A reusable training loop", "Make a deep network train"],
                           3, here_label="← today", title="Where your project stands: your own neural network"),
        mo.md(r"""
        **An everyday example.** A spreadsheet cell with a formula, such as `=A1*B1`, remembers which cells it was made from. Change `A1`, and the spreadsheet knows which cells depend on it.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_grad():
    guess_grad = L.guess(
        "grad",
        question="`a = Value(2)`, `b = Value(3)` and `c = a * b`. You set `c.grad = 1`, and `c` passes its slope back to the values it came from, each times its local gradient. What do you think `a.grad` is then?",
        options={
            "one": "1",
            "product": "6",
            "right": "3",
        },
        reveal="**3.** For $c = a \\cdot b$, the local gradient for $a$ is the number of $b$, 3: when $a$ goes up by 1, $c$ goes up by 3. So $a$ gets $1 \\cdot 3 = 3$, and $b$ gets $1 \\cdot 2 = 2$. `c` can only do that if it remembers `a` and `b`, and those local gradients.",
    )
    guess_grad
    return (guess_grad,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A `Value` holds a number and remembers three things:

        - `data`: its number, from the forward pass;
        - `grad`: the slope of the loss with respect to it, $\frac{\partial \mathcal{L}}{\partial \text{value}}$, which starts at 0;
        - `parents`: the values it was made from, each in a pair with its local gradient.

        An operation on two `Value`s makes a new one, and fills in its `parents` right away, while the numbers are there. `__add__` runs for `+` and `__mul__` for `*`; `relu` is an ordinary method:
        """),
        L.formula(r"\begin{aligned} a + b &: \text{parents } (a, 1),\ (b, 1) \\ a \cdot b &: \text{parents } (a, \text{the number of } b),\ (b, \text{the number of } a) \\ \mathrm{ReLU}(a) &: \text{parent } (a, 1) \text{ if } a > 0, \text{ else } (a, 0) \end{aligned}"),
        mo.md(r"""
        `backward_step` is one step of the backward pass: a value passes its `grad` back to each parent, times the local gradient, and *adds* it to what that parent already has. So the calculation builds the computational graph by itself, and the slopes come back along its arrows. This is **autograd**: automatic gradients. micrograd keeps a small function per value for this step instead of the pairs; the slopes come out the same.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **autograd** | auto-grad | code that builds the computational graph while it calculates, and then works out the slopes by itself | |
        | `Value` | | a number that remembers its slope and where it came from | `Value(2)` |
        | `data`, `grad` | | the number of a `Value`, and the slope of the loss with respect to it | `a.data`, `a.grad` |
        | `parents` | | the values a `Value` was made from, each in a pair with its local gradient | `[(a, 1), (b, 1)]` |
        | `__mul__` | | the special method Python runs for `*`, as `__add__` for `+` | `def __mul__(self, other):` |
        | `backward_step` | | one step of the backward pass: pass `grad` back to each parent, times its local gradient | `c.backward_step()` |
        """),
        L.watch_out("Add to `parent.grad`, do not replace it. A value that goes into two nodes, or twice into one, as `r` in `r * r`, gets a slope back along each arrow, and only their sum is its slope."),
        mo.md(r"""
        Move the slider: it walks through $d = a \cdot b + a$, with $a = 2$ and $b = 3$. Each node shows its `data` and its `grad`; the <span class="c-grad">numbers at the arrows</span> are the local gradients its `parents` hold. A table has the `grad` of every value after each step, one row per step.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_step_cell():
    _say = [
        "Step 0: the forward pass made every `Value`. Each knows its parents and their local gradients, and every `grad` is 0.",
        "Step 1: `d.grad = 1`, the slope of `d` with respect to itself.",
        "Step 2: `d.backward_step()` passes 1 × 1 to `c` and 1 × 1 to `a`: both have `grad` 1.",
        "Step 3: `c.backward_step()` passes 1 × 3 to `a`, which adds it to the 1 it had: 4; and 1 × 2 to `b`: 2. Indeed, $d = ab + a$ has the slopes $b + 1 = 4$ and $a = 2$.",
    ]

    def _frame(step):
        step = int(step)
        g = C.grads(step)
        return (C.picture(step, {"title": "A Value graph, step by step",
                                 "alt": f"Four Value nodes: a = 2 and b = 3 go into c = a · b = 6, with the local gradients 3 and 2; c and a go into d = c + a = 8, with the local gradients 1 and 1. At step {step} of 3 the grads are: a {g[0]}, b {g[1]}, c {g[2]}, d {g[3]}."}).text
                + C.steps_table(step, {"label": "The grads after each step",
                                       "alt": f"A table of the grads of a, b, c and d after each step, one row per step: after step 0 all are 0; after step 1, d is 1; after step 2, a, c and d are 1; after step 3, a is 4, b 2, c 1 and d 1. The row of step {step} is marked."}).text,
                _say[step])

    explore_step = L.explore("step", "step", _frame, start=0, low=0, high=3, step=1, role="", task="Which value gets a slope back twice? What would its `grad` be if `backward_step` replaced `grad` instead of adding to it?")
    explore_step
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    $a$ gets a slope back twice: 1 from $d$ at step 2, and 3 from $c$ at step 3, so its `grad` is 4. If `backward_step` replaced `grad` instead of adding to it, $a$ would keep only the 3 that came last, and its slope would be wrong: 3 instead of 4.

    ### Recipe

    To get slopes with your own `Value`:

    1. Make each number that should get a slope a `Value`.
    2. Calculate with `+`, `*` and `relu()`: each new `Value` fills in its `parents`, with their local gradients.
    3. Set the `grad` of the loss to 1.
    4. Call `backward_step()` on each value, from the loss back, so that a value passes its `grad` on only once it has all of it.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A small `Value` that can only add, on $d = (a + b) + a$. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_value():
    class _Value:
        def __init__(self, data, parents=()):
            self.data = data
            self.grad = 0
            self.parents = parents      # pairs: (the Value it came from, its local gradient)

        def __add__(self, other):
            return _Value(self.data + other.data, [(self, 1), (other, 1)])

        def backward_step(self):
            for parent, local in self.parents:
                parent.grad = parent.grad + local * self.grad

    # 1. the numbers that get a slope
    _a = _Value(2)
    _b = _Value(3)
    # 2. calculate: each new Value remembers where it came from
    _c = _a + _b
    _d = _c + _a
    print(_d.data)
    # 3. the slope of d with respect to itself
    _d.grad = 1
    # 4. from d back: first d, then c
    _d.backward_step()
    _c.backward_step()
    print(_a.grad, _b.grad, _c.grad)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    $d = 2a + b = 7$. `d.backward_step()` gives `c` and `a` each 1. Then `c.backward_step()` gives `a` another 1, so 2, and `b` 1. Those are the slopes of $2a + b$: 2 for $a$ and 1 for $b$.

    Two bits of Python in this class are new. `parents=()` in the `def` line gives `parents` a value for when you leave it out: `_Value(2)` gets the empty tuple `()`, so no parents. And `for parent, local in self.parents:` takes each pair apart, the way `x, y = point` does in [Tuples, sets and sorting](/course/open/py-tuples-sets?lang=en): `parent` gets the first value of the pair, `local` the second.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_order():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. In which order?
    """)
    return


@app.cell(hide_code=True)
def choice_order_cell():
    choice_order = L.choice(
        "order",
        question="In the worked example, what happens if you call `c.backward_step()` before `d.backward_step()`?",
        options={
            "same": "Nothing changes: the slopes come out the same",
            "right": "`b` gets 0, and `a` only 1: `c` passes on its `grad` while it is still 0",
            "error": "Python stops with an error, because `c` was made before `d`",
        },
        correct="right",
        why={
            "same": "At that moment, what is `c.grad`? See step 4 of the recipe.",
            "error": "`backward_step` runs fine on any `Value`; it passes on whatever `grad` it has. What is that for `c` before `d` has passed it anything? See step 4 of the recipe.",
            "right": "`c.grad` is still 0, so `c` passes 0 to `a` and `b`. Then `d` gives `c` its 1, but `c` has already had its turn. That is why the order matters: the next lesson finds it for you.",
        },
    )
    choice_order
    return (choice_order,)


@app.cell(hide_code=True)
def text_exercise_value():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Your Value

        Finish the class `Value`: `__init__` and `__add__` are done, as in the worked example. Replace the three `...`: `__mul__` gives back a new `Value` with the product, and both parents with their local gradients; `relu` gives back a new `Value` with $\mathrm{ReLU}$ of the number, and its one parent with local gradient 1 or 0; `backward_step` passes `grad` back to each parent. The check makes `Value`s, calculates with them and takes backward steps, in a few small graphs. When your class is right, the check saves it in your project, as `work/projects/nn/value.py`.
        """),
        L.task("Give `Value` a `__mul__`, a `relu` and a `backward_step`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_value():
    box_value = L.code_box("value", lines=21)
    box_value
    return (box_value,)


@app.cell(hide_code=True)
def check_value(box_value):
    result_value = L.check("value", box_value.value)
    result_value
    return (result_value,)


@app.cell(hide_code=True)
def hints_value():
    L.hints(
        "value",
        [
            "`__mul__` looks like `__add__`, with the product and other local gradients; `backward_step` is the same as in the worked example.",
            "For $a \\cdot b$, the local gradient for `self` is the number of `other`, and for `other` the number of `self`. ReLU passes its number on above 0, with local gradient 1, and gives 0 elsewhere, with local gradient 0.",
            "`return Value(self.data * other.data, [(self, ___), (other, ___)])`; in `relu`, `if self.data > 0:` with `return Value(self.data, [(self, 1)])`, and otherwise `return Value(0, [(self, ___)])`.",
        ],
        why="`a = Value(2)`, `b = Value(3)`, `c = a * b` makes `c.data` 6, with the parents `(a, 3)` and `(b, 2)`. After `c.grad = 1` and `c.backward_step()`, `a.grad` is 3 and `b.grad` 2. For `r * r`, `r` is both parents, so it gets $r$ times the slope of the product back twice: in total $2r$ times that slope.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_student():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The student, with your Value

        The student of [Backpropagation by hand on a tiny graph](/course/open/nn-backward-by-hand?lang=en): $x = 4$ hours, the grade $y = 7$, at $w = 1$ and $b = 1$. The box imports your `Value` from your project: `from nn.value import Value` takes the class `Value` from the file `value.py` in your package `nn`, as `from math import sqrt` takes `sqrt` from `math`. Then it does the forward pass, with a name for every value. `y + neg` is $y - \hat{y}$: your `Value` cannot subtract, but it can add $\hat{y} \cdot (-1)$. Replace the `...` with the backward pass: the `grad` of the loss, and then one `backward_step()` per value, from the loss back. The last line is `[w.grad, b.grad]`, so the check sees it. The check also tries other numbers for `w` and `b`.
        """),
        L.task("Do the backward pass with your own Value, so that `w.grad` and `b.grad` are the slopes of the loss."),
    ])
    return


@app.cell(hide_code=True)
def exercise_student():
    box_student = L.code_box("student", lines=21)
    box_student
    return (box_student,)


@app.cell(hide_code=True)
def check_student(box_student, result_value):
    result_value  # check again when exercise 2 saves your module
    result_student = L.check("student", box_student.value)
    result_student
    return (result_student,)


@app.cell(hide_code=True)
def hints_student():
    L.hints(
        "student",
        [
            "Steps 3 and 4 of the recipe: first the `grad` of `loss`, then the values from the last one made back to the first.",
            "Each value passes its `grad` on once it has all of it. Which value was made last, and which just before it?",
            "`loss.grad = 1`, then `loss.backward_step()`, `r.backward_step()`, `neg.backward_step()`, `___.backward_step()` and `___.backward_step()`.",
        ],
        why="From the loss back: `loss`, `r`, `neg`, `y_pred` and `m`. `loss = r * r` gives `r` the slope $2 \\cdot 2 = 4$, `neg` passes $4 \\cdot (-1) = -4$ on to `y_pred`, and `y_pred` gives `m` and `b` each −4. `m` gives `w` $-4 \\cdot 4 = -16$: the slopes −16 and −4 of the lesson before, now from your own `Value`.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell below trains $w$ and $b$ for five students with your own `Value`: for each student, the same graph as in exercise 3, a backward pass in the same order, and then one step of gradient descent on the slopes of all five together, 300 times. The cell uses your own module, so it works once exercise 2 is done.
    """)
    return


@app.cell(hide_code=True)
def demo_train(result_value):
    result_value  # run again when exercise 2 saves your module
    _fit = C.trained()
    if _fit is None:
        _out = mo.md("*Finish exercise 2 first: then your `Value` is in your project, and this cell uses it.*")
    elif not _fit:
        _out = mo.md("*The `Value` in `work/projects/nn/value.py` no longer gives the right slopes, or stops with an error. Did you change that file? Delete it, and check exercise 2 again: its check then saves your class again.*")
    else:
        _w, _b, _before, _after = _fit
        _out = mo.vstack([
            mo.Html(C.fit_picture(_w, _b, {"title": "Five students, and the line your Value trained", "x": "hours studied", "y": "grade",
                                           "alt": f"Five points, the hours and grades of five students, and a straight line through them with w = {L.number(_w, 2)} and b = {L.number(_b, 2)}."})),
            mo.md(f"""Your `Value` found $w$ = <span class="c-param ea-num">{L.number(_w, 2)}</span> and $b$ = <span class="c-param ea-num">{L.number(_b, 2)}</span>, and the mean squared error went from <span class="c-loss ea-num">{L.number(_before, 2)}</span> to <span class="c-loss ea-num">{L.number(_after, 2)}</span>. Every slope came from your own `backward_step`; the order of the steps still came from the course. In the next lesson your project finds that order itself."""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Backpropagation by hand on a tiny graph](/course/open/nn-backward-by-hand?lang=en) you kept the record of the graph yourself; now each `Value` keeps it, with the special methods of [Special methods: making + and print work on your objects](/course/open/py-special-methods?lang=en). Your project `nn` now has `value.py`. In the next lesson, *The full backward pass, in the right order*, your project gets a `backward()` that finds the order of the steps by itself, for any graph.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_order, result_value, result_student):
    L.wrap_up(choice_order, result_value, result_student)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
