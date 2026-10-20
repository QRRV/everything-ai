# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-quantize. Route: course/route/lessen/taal.yaml. Checks and the measured numbers: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import torch
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-quantize", "en")
    C = L.checks
    M = C.MEASURED            # the course's model, measured: its loss on the stories at each number of bits
    SIZE = C.course_sizes()   # and its megabytes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Fewer bits: FP32, FP16 and BF16](/course/open/sys-number-formats?lang=en) and [Capstone: pretrain your own small LLM](/course/open/llm-pretrain-capstone?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bf16():
    quick_bf16 = L.choice(
        "quick-bf16",
        question="BF16 and FP16 both store a number in 16 bits. What does BF16 give up, compared with FP16?",
        options={
            "range": "Range: its numbers cannot get as large",
            "right": "Mantissa bits: it keeps fewer digits",
            "sign": "The sign bit: it stores only numbers of 0 and more",
        },
        correct="right",
        why={
            "right": "BF16 spends 8 bits on its exponent, as FP32 does, and keeps 7 mantissa bits where FP16 keeps 10: the same range as FP32, with fewer digits.",
            "range": "That is FP16: with 5 exponent bits it overflows above 65,504. BF16 keeps the 8 exponent bits of FP32. See [Fewer bits: FP32, FP16 and BF16](/course/open/sys-number-formats?lang=en).",
            "sign": "All three formats keep a sign bit: weights and gradients can be negative. See [Fewer bits: FP32, FP16 and BF16](/course/open/sys-number-formats?lang=en).",
        },
    )
    quick_bf16
    return (quick_bf16,)


@app.cell(hide_code=True)
def quick_check_megabytes():
    quick_megabytes = L.choice(
        "quick-megabytes",
        question="The course's model of the capstone has 3,141,801 parameters, each stored in FP32. About how many megabytes do they take?",
        options={
            "right": "12.6 MB: 4 for each parameter",
            "bits": "100.5 MB: 32 for each parameter",
            "one": "3.1 MB: 1 for each parameter",
        },
        correct="right",
        why={
            "right": "FP32 is 32 bits, 4 bytes, per number: 3,141,801 × 4 bytes is 12.6 million bytes.",
            "bits": "32 is the bits of an FP32 number; a byte holds 8 bits, so FP32 is 4 bytes. See [Fewer bits: FP32, FP16 and BF16](/course/open/sys-number-formats?lang=en).",
            "one": "One byte is 8 bits, and FP32 stores each number in 32: 4 bytes per parameter. See [Fewer bits: FP32, FP16 and BF16](/course/open/sys-number-formats?lang=en).",
        },
    )
    quick_megabytes
    return (quick_megabytes,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    To run a model, a computer keeps its weights in memory, and for every token it writes it reads all of them once. An open model of 7 billion parameters takes 14 GB in BF16: more than most laptops can spare, and more than many GPUs hold. And reading 14 GB for every token is slow: as [Memory or compute: what limits your GPU?](/course/open/sys-memory-roofline?lang=en) showed, a model that writes one token at a time is memory-bound, with about two operations for every weight it reads. Store every weight in 4 bits instead of 16, and the model takes about a quarter of the memory, and there is about a quarter as much to read for every token. The price is precision: how much that costs, you measure.

    **An everyday example.** A photo from your phone, saved as a JPEG for a website: ten times smaller, and you have to look closely to see what it lost. Save it too small, and it falls apart into blocks.
    """)
    return


@app.cell(hide_code=True)
def predict_loss():
    guess_loss = L.guess(
        "loss",
        question=f"The course's model of the capstone has a loss of {C.number(M[(32, None)], 2, 'en')} on stories it never trained on. Stored in 4 bits per weight instead of 32, the way open models are usually shared, it is about seven times smaller. What happens to that loss?",
        options={
            "small": "It goes up a little, by a few hundredths",
            "half": "It goes up by about 0.5, to above 3",
            "broken": "It goes above 6: the model writes nonsense",
        },
        reveal=f"**It goes up a little:** to {C.number(M[(4, C.GROUP)], 2, 'en')}, when every group of {C.GROUP} weights gets its own scale. The model shrinks from {C.number(SIZE[(32, None)], 1, 'en')} MB to {C.number(SIZE[(4, C.GROUP)], 1, 'en')} MB. But the other two answers happen as well, stored another way: with one scale for a whole matrix, the loss goes to {C.number(M[(4, 'matrix')], 2, 'en')}, and in 2 bits to {C.number(M[(2, C.GROUP)], 2, 'en')}. How you store the bits matters as much as how many there are.",
    )
    guess_loss
    return (guess_loss,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A grid of whole numbers.** With $b$ bits you can store $2^b$ different whole numbers. To keep 0 in the middle, this kind of quantization uses the whole numbers from $-q_{max}$ to $q_{max}$, with $q_{max} = 2^{b-1} - 1$: from −7 to 7 in 4 bits, from −127 to 127 in 8. Each weight $w$ becomes the whole number $q$ nearest to $w / s$, where $s$ is a scale that maps the grid onto the weights:
        """),
        L.formula(r"s = \frac{\max |w|}{q_{max}}, \qquad q = \mathrm{round}\left(\frac{w}{s}\right), \qquad \hat w = q \cdot s"),
        mo.md(r"""
        The weight largest in size lands exactly on $\pm q_{max}$, and every other one at most half a step $s/2$ from where it was. What is stored is $q$, in $b$ bits, and $s$; the model computes with $\hat w = q \cdot s$.

        **Groups.** One large weight stretches the grid for all the weights that share its scale, and the small ones round to 0. So each row of a matrix is cut into groups of, say, 32 weights, and each group gets its own scale. A scale takes 16 bits, so for groups of 32 it adds half a bit per weight: 4 bits become 4.5.

        **Measure, do not assume.** A small error per weight says little about the model: some weights matter more than others. Its loss on text it never trained on, before and after, says what it cost.

        ### New words and symbols

        | Word or symbol | What it means |
        |---|---|
        | **quantization** | storing numbers, here a model's weights, as whole numbers of a few bits with a scale, so that they take less memory |
        | **scale** ($s$) | the number that the whole numbers of a group are multiplied by to give its weights back: its largest $\lvert w \rvert$ divided by $q_{max}$ |
        | **group** | the weights that share one scale, such as 32 weights next to each other in a row of a matrix |
        | $q_{max}$ | the largest whole number of $b$ bits with as many levels below 0 as above it: $2^{b-1} - 1$ |
        | `t.amax(dim=1, keepdim=True)` | the largest number of `t` along axis 1, kept as an axis of length 1 |
        | `torch.round(t)` | every number of `t` rounded to the nearest whole number |
        | `t.clamp(lo, hi)` | every number of `t` kept between `lo` and `hi` |

        The picture shows one group of 32 weights of the course's model, from the MLP of its first block: the rings are the weights as they are, the dots where they land in $b$ bits, and the thin lines the grid of levels. Move the slider up to 8 bits and down to 2.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_bits_cell():
    _labels = {"title": "One group of 32 weights, in fewer bits", "x": "place in the group", "y": "value",
               "alt": "The 32 weights of one group of the course's model as rings, and the values they get in the chosen number of bits as dots on a grid of levels."}

    def _frame(b):
        _b = int(b)
        _n = C.group_numbers(_b)
        _say = (f"**{_b} bits:** {_n['levels']} levels, a step of {C.number(_n['step'], 4, 'en')}; "
                f"the largest error is {C.number(_n['error'], 4, 'en')}, under half a step.")
        return C.group_picture(_b, _labels), _say

    explore_bits = L.explore("bits", "Bits per weight", _frame, start=4, low=2, high=8, step=1)
    explore_bits
    return (explore_bits,)


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        At 8 bits the dots sit on the rings: 255 levels, too close to draw. At 4 bits the 15 levels show, and every dot has moved to its nearest line. At 2 bits there are three levels, $-s$, 0 and $s$: 22 of the 32 weights become 0, and only the 10 largest keep their sign.
        """),
        L.watch_out("The whole numbers save memory; they do not make the arithmetic cheaper by themselves. With weights in 4 bits, the computer usually turns each group back into 16-bit numbers, $q \\cdot s$, to multiply, in its fast memory near the cores. The gain is in what it reads from the slow memory."),
        mo.md(r"""
        ### Recipe

        To store a model in fewer bits:

        1. Choose the bits $b$ and the group size, such as 4 bits and groups of 32.
        2. For every matrix of the model, cut each row into groups, and give each group its scale: its largest $\lvert w \rvert$ divided by $q_{max} = 2^{b-1} - 1$.
        3. Round every weight to its whole number $q = \mathrm{round}(w / s)$, from $-q_{max}$ to $q_{max}$; store $q$ in $b$ bits and the scales in 16.
        4. Measure the model's loss on text it never trained on, before and after.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Four weights as one row and one group, in 4 bits, in PyTorch. Run the cell.
    """)
    return


@app.cell
def example_group():
    _w = torch.tensor([[0.12, -0.5, 0.31, 0.02]])  # one group
    _qmax = 2 ** (4 - 1) - 1  # 4 bits: -7 to 7
    # the largest |w| lands on 7
    _scale = _w.abs().amax(dim=1, keepdim=True) / _qmax
    _q = torch.round(_w / _scale).clamp(-_qmax, _qmax)
    print("scale:", round(_scale.item(), 4))
    print("q:", _q.tolist())
    print("back:", [round(v, 4) for v in (_q * _scale)[0].tolist()])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The scale is $0.5 / 7 = 0.0714$. The weight largest in size, $-0.5$, becomes $-7$ and comes back exactly; $0.12$ becomes 2, which gives back $0.1429$; and $0.02$ is less than half a step, so it becomes 0. Stored, the group is four whole numbers of 4 bits and one scale of 16: 32 bits instead of 128.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_quantize():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Quantize a matrix

        Complete `quantize(w, bits, group)`: the weights of the matrix `w` as the model computes with them after storing each in `bits` bits, with a scale per group of `group` weights of a row. The first lines already cut each row into groups: `groups` has the shape (rows, groups per row, `group`). `groups.abs().amax(dim=2, keepdim=True)` gives the largest $\lvert w \rvert$ of every group, `torch.round` rounds every number, and `.clamp(-qmax, qmax)` keeps the whole numbers in their range. A group of only zeros has a scale of 0: give it a scale of 1, and its weights stay 0.

        The check tries an example, a matrix of two rows, in 4 bits with groups of 2; then one scale per row, 2 and 8 bits, a group of zeros, and the shape of the result.
        """),
        L.task("Complete `quantize`: the scale of every group, the whole numbers, and the weights back in the shape of `w`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_quantize():
    box_quantize = L.code_box("quantize", lines=16)
    box_quantize
    return (box_quantize,)


@app.cell(hide_code=True)
def check_quantize(box_quantize):
    result_quantize = L.check("quantize", box_quantize.value)
    result_quantize
    return (result_quantize,)


@app.cell(hide_code=True)
def hints_quantize():
    L.hints(
        "quantize",
        [
            "The scale of every group is its largest $\\lvert w \\rvert$ divided by `qmax`; `amax(dim=2, keepdim=True)` keeps one number per group, in a shape that divides `groups` at once.",
            "`q` is `groups / scale`, rounded and clamped to `-qmax` and `qmax`; what the model computes with is `q * scale`, reshaped to `(rows, cols)`.",
        ],
        why="Each group gets the scale that puts its largest weight on $\\pm q_{max}$; rounding to whole numbers is the storing in few bits, and `q * scale` is what the model computes with.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(rf"""
    ### What your model does in fewer bits

    The button stores every matrix of your model of the capstone (or the course's, if you have none) with your `quantize`, in 8, 4, 3 and 2 bits with groups of {C.GROUP}, and in 4 bits with one scale per row. For each it measures the size and the loss on {C.number(C.TOKENS, 0, 'en')} tokens of stories the model never trained on, and it lets the model write a story in 4 bits and in 2. That takes about a minute on a laptop.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    demo_button = mo.ui.run_button(label="Store my model in fewer bits")
    demo_button
    return (demo_button,)


@app.cell(hide_code=True)
def demo_quantize(box_quantize, demo_button, result_quantize):
    if result_quantize.state != "correct":
        _out = mo.md("*Make `quantize` right in exercise 1 first: the demo stores your model with it.*")
    elif not demo_button.value:
        _out = mo.md("*Press the button to store your model in fewer bits.*")
    else:
        _r = C.demo(box_quantize.value, lambda _b, _g: mo.output.replace(mo.md(f"**Measuring** {_b} bits …")))
        if _r == "run":
            _out = mo.md("*Loading your model stops with an error. Do your modules in `work/projects/llm/` still pass their checks? Delete a file you changed, and check its exercise again.*")
        elif _r == "quantize":
            _out = mo.md("*Your `quantize` stops with an error on a matrix of the model, or gives back weights of another shape or that are not numbers. Does it work for a matrix of hundreds of rows?*")
        else:
            _base = _r["rows"][0]["loss"]
            _name = lambda _x: "32 (as it is)" if _x["bits"] == 32 else str(_x["bits"])
            _grp = lambda _x: "" if _x["group"] is None else ("a row" if _x["group"] == "row" else str(_x["group"]))
            _table = "| bits | group | size | loss on the stories | change |\n|---|---|---|---|---|\n" + "\n".join(
                f"| {_name(_x)} | {_grp(_x)} | {C.number(_x['bytes'] / 1e6, 1, 'en')}\u00a0MB | {C.number(_x['loss'], 2, 'en')} | "
                f"{'' if _x['bits'] == 32 else ('+' if _x['loss'] >= _base else '−') + C.number(abs(_x['loss'] - _base), 2, 'en')} |" for _x in _r["rows"])
            _out = mo.vstack([
                mo.md(f"**{'Your model' if _r['source'] == 'yours' else 'The course’s model'}**, {C.number(_r['params'], 0, 'en')} parameters:\n\n" + _table),
                mo.Html(C.demo_picture(_r["rows"], {"title": "Loss on the stories, groups of 32", "x": "bits per weight", "y": "loss",
                                                    "full": "as it is", "alt": "The loss on the stories against the bits per weight, with the model as it is as a dashed line."})),
                mo.md("**In 4 bits** it writes:\n\n> " + _r["stories"][4].replace("\n", " ") + "\n\n**In 2 bits:**\n\n> " + _r["stories"][2].replace("\n", " ")),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_memory():
    mo.md(r"""
    ### Exercise 2. An open model on your laptop
    """)
    return


@app.cell(hide_code=True)
def choice_memory_cell():
    choice_memory = L.choice(
        "memory",
        question="An open model of 7 billion parameters takes 14 GB in BF16. You store its weights in 4 bits, with a 16-bit scale for every group of 32 weights. About how big is it then?",
        options={
            "quarter": "3.5 GB: a quarter of 14",
            "half": "7 GB: half of 14",
            "right": "3.9 GB: 4.5 bits per weight",
        },
        correct="right",
        why={
            "right": "Each weight takes 4 bits, and each group of 32 adds a scale of 16 bits, half a bit per weight: 7 billion × 4.5 bits is 31.5 billion bits, 3.9 GB.",
            "quarter": "4 bits is a quarter of 16, but every group of 32 also stores its scale of 16 bits: half a bit more per weight. Look at *Groups* in the idea again.",
            "half": "Half would be 8 bits per weight. In 4 bits, with half a bit for the scales, it is 4.5 bits instead of 16.",
        },
    )
    choice_memory
    return (choice_memory,)


@app.cell(hide_code=True)
def text_exercise_faster():
    mo.md(r"""
    ### Exercise 3. Why it writes faster
    """)
    return


@app.cell(hide_code=True)
def explain_faster():
    explain_faster_cell = L.explain(
        "faster",
        question="On your laptop, the model in 4 bits writes its text faster than the same model in BF16, although each multiplication is still done in 16 bits. Why?",
        model_answer="Writing one token reads every weight of the model once from memory, and does only about two operations with each: it is memory-bound, so its time is about the bytes it reads divided by how fast memory delivers them. In 4 bits, with half a bit for the scales, there is about 4.5 / 16, a bit more than a quarter, as much to read. Turning each group back into 16-bit numbers happens in the fast memory near the cores, where it costs little. So the model can write up to about three and a half times as fast, at the small cost in loss I measured.",
        points=[
            "writing a token reads every weight once: memory-bound",
            "fewer bits means fewer bytes to read for every token",
            "the multiplications themselves are not what takes the time",
        ],
    )
    explain_faster_cell
    return (explain_faster_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Fewer bits: FP32, FP16 and BF16](/course/open/sys-number-formats?lang=en) stored every number as a float of its own, with its own exponent; quantization stores whole numbers that share one scale per group, which is why it can go down to 4 bits. The model is the one of [Capstone: pretrain your own small LLM](/course/open/llm-pretrain-capstone?lang=en), and its loss on stories it never saw is how you see what fewer bits cost. Open models are shared in 4 or 8 bits in much this way, so that they fit on a laptop.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_quantize, choice_memory, explain_faster_cell):
    L.wrap_up(result_quantize, choice_memory, explain_faster_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
