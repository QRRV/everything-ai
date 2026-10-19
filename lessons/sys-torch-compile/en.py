# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-torch-compile. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.
# termen:allow graph

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-torch-compile", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Write your own GPU kernel](/course/open/sys-triton-kernel?lang=en) and [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kernel():
    quick_kernel = L.choice(
        "quick-kernel",
        question="What is a GPU kernel?",
        options={
            "core": "One of the cores of a GPU",
            "right": "A function that runs on the GPU, in many programs side by side",
            "driver": "The driver that lets PyTorch talk to the GPU",
        },
        correct="right",
        why={
            "core": "A kernel is code, not hardware: one launch runs it in many programs over the cores. See [Write your own GPU kernel](/course/open/sys-triton-kernel?lang=en).",
            "driver": "A kernel is the code of one operation, such as your fused softmax. See [Write your own GPU kernel](/course/open/sys-triton-kernel?lang=en).",
            "right": "Every PyTorch operation that computes on a GPU launches at least one. This lesson lets a compiler write fused ones for you.",
        },
    )
    quick_kernel
    return (quick_kernel,)


@app.cell(hide_code=True)
def quick_check_bound():
    quick_bound = L.choice(
        "quick-bound",
        question="What limits the speed of `x * 2 + 1` on a tensor of 100 million numbers?",
        options={
            "right": "The memory traffic: reading and writing every number",
            "flops": "The FLOPs: two per number",
            "launch": "Starting the kernel",
        },
        correct="right",
        why={
            "flops": "Two FLOPs per 8 bytes moved is far below what a GPU or CPU computes per byte. See [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en).",
            "launch": "Starting a kernel takes microseconds; moving 800 MB takes far longer. See [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en).",
            "right": "It is memory-bound. Run as two operations, it even sends the tensor through memory twice: fusing them is what a compiler does.",
        },
    )
    quick_bound
    return (quick_bound,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Your fused softmax was faster than the five PyTorch operations, but writing a kernel for every chain of operations in a model is a lot of work. **`torch.compile`** does it for you: one line, and PyTorch traces your model, fuses what it can, and writes the kernels. It can make training clearly faster, and it can also gain nothing. Knowing what it does tells you when to use it, and how to measure whether it helped.
    """)
    return


@app.cell(hide_code=True)
def predict_first():
    guess_first = L.guess(
        "first-call",
        question="You compile a small function with `torch.compile`, and call it once. Compared with the same call without compiling, that one call is…",
        options={
            "fast": "Faster: compiled code is fast",
            "same": "About as fast",
            "right": "Much slower",
        },
        reveal=f"**Much slower:** the first call compiles. For a function of three elementwise operations and a sum on 4 million numbers, on the CPU the course was made on, that took about {L.number(C.COMPILE_S, 0)} seconds; after that, a call took {L.number(C.COMPILED_S * 1000, 1)} ms instead of {L.number(C.EAGER_S * 1000, 1)} ms. Compiling pays off only when the function runs often.",
    )
    guess_first
    return (guess_first,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Step 1: trace.** At the first call, **Dynamo**, the front end of `torch.compile`, reads your Python function as it runs, and records the PyTorch operations into a graph. It adds **guards**: checks on the inputs, such as their shapes and number types. At later calls, while the guards hold, it reuses the compiled code; when one fails, for example with a new shape, it compiles again.

        **Step 2: compile.** **Inductor**, the back end, takes the graph, fuses the operations that can go together, and writes the kernels: in Triton on an NVIDIA GPU, in C++ on a CPU. That is the slow part of the first call.

        **Step 3: graph breaks.** What Dynamo cannot put in a graph, it leaves to Python: a `print`, `.item()` that takes a number out of a tensor, an `if` on the value of a tensor, a library that is not PyTorch. There the graph breaks in two, and nothing is fused across the break. `torch._dynamo.explain(fn)(x)` traces a function, without compiling, and counts the graphs and the breaks.

        **Step 4: does it pay off?** A compiled call is faster by $t_\text{eager} - t_\text{compiled}$; compiling costs $t_\text{compile}$ once. It has paid for itself after
        """),
        L.formula(r"n^* = \left\lceil \frac{t_\text{compile}}{t_\text{eager} - t_\text{compiled}} \right\rceil \text{ calls}"),
        mo.md(r"""
        and never when a compiled call is not faster. A training run of 100,000 steps of the same shape pays off easily; a script that runs once does not.

        The picture: the total time against the number of calls, for the measurement of the guess: without compiling, a line through 0; compiled, a line that starts at the compile time but rises less steeply. They cross at $n^*$.
        """),
        mo.Html(C.calls_picture({"title": "When compiling pays off", "x": "calls", "y": "total seconds",
                                 "eager": "without compiling", "compiled": "compiled", "even": "n* ≈ {n}", "lang": L.lang,
                                 "alt": "Two lines of total time against 0 to 2,100 calls. Without compiling, dashed, it rises from 0 to about 47 seconds. Compiled, solid, it starts at 16 seconds and rises to about 22. They cross at about 800 calls."})),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **`torch.compile`** | | traces a model or function and compiles it into fused kernels, at its first call | `torch.compile(model)` |
        | **graph break** | | a place where tracing stops and Python takes over, splitting the graph in two | `torch._dynamo.explain(fn)(x)` |
        | **guard** | | a check on the inputs under which the compiled code may be reused | |
        | `torch.where` | torch where | number by number: `a` where the condition `c` holds, else `b`; one graph, no `if` | `torch.where(c, a, b)` |
        | $n^*$ | n star | the number of calls after which compiling has paid for itself | |
        """),
        L.watch_out("Measure a compiled function only after its first call, and on a GPU with `torch.cuda.synchronize()` before you read the clock, as in [Write your own GPU kernel](/course/open/sys-triton-kernel?lang=en). And watch for recompiles: a new input shape fails a guard. A model that gets texts of every length compiles again at the second length, this time for any length, and code for any length can be slower than code for one; pad to a few fixed lengths, or pass `dynamic=True` to compile for any length at the first call."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find out whether `torch.compile` helps:

    1. Count the graph breaks with `torch._dynamo.explain`, and remove the ones in the hot path.
    2. Time a call without compiling, after a warmup call.
    3. Compile, time the first call, which compiles, and then time later calls.
    4. Work out $n^*$, and compare it with how often the code will run.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 1 on a function with a `print` in the middle: `torch._dynamo.explain` traces it, without compiling, and tells how many graphs it got and why it broke.
    """)
    return


@app.cell
def example_explain():
    import torch
    import torch._dynamo as dynamo

    def _halves(x):
        _y = torch.relu(x) * 2
        print("halfway")
        return _y + 1

    _e = dynamo.explain(_halves)(torch.tensor([-1.0, 2.0]))
    print("graphs:", _e.graph_count, "breaks:", _e.graph_break_count)
    _why = str(_e.break_reasons[0].reason).splitlines()
    print(_why[0])
    print(_why[1])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `explain` runs the function, so "halfway" is printed. Dynamo could not put the `print` in a graph, as the second line of the reason says: it made one graph before it and one after, with one break. The ReLU and the times 2 can still be fused, and the plus 1 stands alone; across the break, nothing is fused. Without the `print`, it is one graph.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_even():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. When it pays off

        Complete `break_even(compile_s, eager_s, compiled_s)`: with the compile time and the seconds of one call without and with compiling, give back $n^*$, rounded up, or `None` when a compiled call is not faster. The check calls `break_even(30.0, 0.05, 0.02)`, which should give `1000`, and tries the measurement of the picture, a division that comes out whole, equal times, and a slower compiled call.
        """),
        L.task("Make `break_even` give the number of calls after which compiling has paid off."),
    ])
    return


@app.cell(hide_code=True)
def exercise_even():
    box_even = L.code_box("break-even", lines=8)
    box_even
    return (box_even,)


@app.cell(hide_code=True)
def check_even(box_even):
    result_even = L.check("break-even", box_even.value)
    result_even
    return (result_even,)


@app.cell(hide_code=True)
def hints_even():
    L.hints(
        "break-even",
        [
            "First the case where a compiled call is not faster; then the formula of Step 4, with `math.ceil` to round up.",
        ],
        why="`if compiled_s >= eager_s: return None`, and then `return math.ceil(compile_s / (eager_s - compiled_s))`. For the example: 30 / 0.03 = 1,000 calls.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_graph():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. One graph

        `one_graph(x)` divides `x` by its largest absolute value when that is above 1, and takes the ReLU. As written, it breaks the graph: `.item()` takes the maximum out of the graph as a Python number, and the `if` then decides in Python. Rewrite it so that it gives the same outcome, as one graph: keep the maximum a tensor, and let `torch.where(condition, a, b)` choose: it takes `a` where the condition holds, and `b` elsewhere. The check calls `one_graph(torch.tensor([0.5, -2.0, 1.5, 3.0]))`, which should give `[0.1667, 0.0, 0.5, 1.0]`, tries other tensors, and counts the graph breaks with `torch._dynamo.explain`.
        """),
        L.task("Make `one_graph` give the same outcome without a graph break."),
    ])
    return


@app.cell(hide_code=True)
def exercise_graph():
    box_graph = L.code_box("one-graph", lines=9)
    box_graph
    return (box_graph,)


@app.cell(hide_code=True)
def check_graph(box_graph):
    result_graph = L.check("one-graph", box_graph.value)
    result_graph
    return (result_graph,)


@app.cell(hide_code=True)
def hints_graph():
    L.hints(
        "one-graph",
        [
            "`m = x.abs().max()` stays a tensor with one number; `m > 1` is then a tensor too, and `torch.where` takes it as its condition.",
        ],
        why="`m = x.abs().max()` and `return torch.relu(torch.where(m > 1, x / m, x))`. For the example, the maximum is 3: the vector is divided by 3, and the ReLU makes −0.667 zero.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Compile and time

    The button compiles your `one_graph` once it is right, else the course's, and times it on 4 million numbers, on your NVIDIA GPU if you have one: the first call, which compiles, and then a call without and with compiling. Compiling can take up to a minute; PyTorch keeps what it compiled in a cache on disk, so the next time the first call is much faster. On a computer without a C++ compiler it does not work.
    """)
    return


@app.cell(hide_code=True)
def compile_button_cell():
    compile_button = mo.ui.run_button(label="Compile and time")
    compile_button
    return (compile_button,)


@app.cell(hide_code=True)
def demo_compile(box_graph, compile_button, result_graph):
    if not compile_button.value:
        _out = mo.md("*Press the button to compile and time.*")
    else:
        _d = C.demo(box_graph.value if result_graph.state == "correct" else None)
        if _d == "error":
            _out = mo.md("*Compiling did not work on this computer: `torch.compile` needs a C++ compiler on a CPU, or an NVIDIA GPU with Triton.*")
        else:
            _where = "your GPU" if _d["device"] == "cuda" else "your CPU"
            _who = "your `one_graph`" if _d["yours"] else "the course's `one_graph`"
            _gain = _d["eager"] / _d["compiled"]
            _out = mo.md(
                f"On {_where}, {_who}: the first call took {L.number(_d['first'], 1)} s; then a call took "
                f"{L.number(_d['eager'] * 1000, 2)} ms without compiling and {L.number(_d['compiled'] * 1000, 2)} ms compiled, "
                f"{L.number(_gain, 1)} times as fast. The outcomes are {'the same' if _d['same'] else 'different'}."
            )
    _out
    return


@app.cell(hide_code=True)
def text_exercise_when():
    mo.md(r"""
    ### Exercise 3. When to compile
    """)
    return


@app.cell(hide_code=True)
def choice_when_cell():
    choice_when = L.choice(
        "when-compile",
        question="Which of these gains most from `torch.compile`?",
        options={
            "once": "A script that runs a model once on one input",
            "right": "A training run of 100,000 steps on batches of one shape",
            "lengths": "A loop that gets a text of another length at every call",
        },
        correct="right",
        why={
            "once": "One call never makes up for the compile time: $n^*$ is far above 1.",
            "lengths": "A new length fails a guard, so it compiles a second time, now for any length: an extra compile, and code for any length can be slower than code for one shape.",
            "right": "One compile, then 100,000 calls with the same guards: far past $n^*$.",
        },
    )
    choice_when
    return (choice_when,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The fusing you did by hand in [Write your own GPU kernel](/course/open/sys-triton-kernel?lang=en), now done by a compiler, which writes Triton kernels of its own on an NVIDIA GPU; what it saves is the memory traffic of [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en). In a training run of your own, compile the model once, check the graph breaks, and time a few steps after the first.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_when, result_even, result_graph):
    L.wrap_up(result_even, result_graph, choice_when)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
