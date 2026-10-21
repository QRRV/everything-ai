# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-reasoning-from-scratch. Route: course/route/lessen/taal.yaml. Checks and the course's runs: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-reasoning-from-scratch", "en")
    C = L.checks
    M = C.MEASURED["cpu"]   # the course's three seeds of the CPU plan: right of 200 per run and length
    K = C.course_result("cpu")   # the course's runs of seed 0 on the CPU plan


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en) and [How an LLM reasons: written steps and test-time compute](/course/open/llm-reasoning-idea?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mask():
    _quick_mask = L.choice(
        "quick-mask",
        question="In SFT, the labels of the instruction's tokens are -100. Why?",
        options={
            "right": "So that the loss counts only the answer",
            "hidden": "So that the model never sees the instruction",
            "padding": "To mark where the padding begins",
        },
        correct="right",
        why={
            "right": "`F.cross_entropy` skips every label of -100: the model reads the instruction, and learns to write only the answer. Today the prompt is a multiplication.",
            "hidden": "The model does read the instruction: it is its input. Only the loss leaves it out. See [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en).",
            "padding": "The padding of the targets gets -100 too, but the instruction gets it for another reason: the loss should count only the answer. See [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en).",
        },
    )
    _quick_mask
    return


@app.cell(hide_code=True)
def quick_check_steps():
    _quick_steps = L.choice(
        "quick-steps",
        question="Why can written steps help a model of fixed depth to answer?",
        options={
            "right": "Every token it writes is one more forward pass, and each step can be a problem that one pass can do",
            "weights": "Writing the steps changes its weights while it answers",
            "copy": "The steps hold the answer, so the model only has to copy it",
        },
        correct="right",
        why={
            "right": "More computation for the same question, in small pieces. Today you test that on a model whose every training example you choose.",
            "weights": "No weight changes while a model answers: the steps are text it writes and then reads. See [How an LLM reasons](/course/open/llm-reasoning-idea?lang=en).",
            "copy": "The model writes the steps itself; they only hold the answer once it has worked it out. See [How an LLM reasons](/course/open/llm-reasoning-idea?lang=en).",
        },
    )
    _quick_steps
    return


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Qwen3 answered better with written steps, but it was pretrained on trillions of tokens, probably with worked solutions among them: you cannot tell what the steps did there. So test it where you choose every example: train the same small model from scratch twice on the same multiplications, once on the answer alone and once with the steps written out. Only the format of the data differs: a fair comparison, as in [Fair comparisons](/course/open/res-fair-baselines?lang=en), with the answer alone as the control run.

        The model is new and small, built only from parts of your project: ModernGPT, `sft_example`, `sft_batch` and `pretrain_step`. Your story model stays as it is.
        """),
        draw.project_steps(["Count letters and pairs", "A bigram model by counting", "The same model, learned by a neural network",
                            "Your own tokenizer", "Your own GPT", "Pretrain a small LLM", "Teach it to follow instructions and preferences",
                            "Extra: a small model that writes its steps"],
                           7, here_label="← today", title="Where your project stands: your own LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_formats():
    _s, _p = M["steps-1800"]["3"], M["plain-1800"]["3"]
    _guess_formats = L.guess(
        "formats",
        question=f"The same model, the same {C.number(C.RUNS['cpu'][0][1], 0, 'en')} steps of {C.BATCH} pairs, drawn the same way. On {C.HELD} products of two 3-digit numbers it never saw, how often does each get it right?",
        options={
            "none": "Both hardly ever: 3-digit products are too hard for a model this small",
            "both": "Both about as often: they saw the same pairs",
            "steps": "With steps most of them, the answer alone hardly any",
        },
        reveal=f"**With steps {C.listed(_s, 'en')} of {C.HELD}; the answer alone {C.listed(_p, 'en')}**, in the course's runs with seeds 0, 1 and 2. The same model and the same pairs: only the format of the answers changed.",
    )
    _guess_formats
    return


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(rf"""
        ## The idea

        **Training on written steps.** The answers in the training data hold the steps of their computation, so the model learns to write them before the result: a chain of thought, as training data. Each step is a small problem, and every token the model writes is a forward pass that reads the steps before it.

        **The two formats.** The tokens are characters: the ten digits, `*`, `+`, `=`, `,` and `>`, and the end-of-text, {C.SIZES[0]} in all. The prompt is `17*21=`, the loss counts only the answer (loss masking, as in SFT), and the answer ends with the end-of-text.

        - answer only: `357`
        - with steps: `17*1=17,17*2=34>17+340=357>357`. First the partial products, 17 times each digit of 21, from the right; then `>` and the running sums, with the place of each digit: 17 + 340; then `>` and the product. With a `b` of one digit there are no sums: `7*1=7>>7`.

        ### New words and symbols

        | Word or symbol | What it means |
        |---|---|
        | **training on written steps** | training on answers that hold the steps of their computation, so that the model learns to write them before the result |
        | **length generalization** | whether a model trained on inputs up to some length also gets longer ones right |
        """),
        L.watch_out("*Writes its steps* is not *reasons*. This model learns one computation in one format; a reasoning model, such as OpenAI o1 (as of October 2026), writes long chains of thought about many kinds of problems."),
    ])
    return


@app.cell(hide_code=True)
def text_play_formats():
    mo.md(r"""
    **Both formats, token by token.** Grey is the prompt, which the model is given; blue is what it learns to write, where the loss counts. The slider picks a pair, from 1 × 1 digits to 4 × 4.
    """)
    return


@app.cell(hide_code=True)
def play_formats():
    _labels = {"title": "{a} × {b}: the tokens of each format", "plain": "answer only", "steps": "with steps",
               "tokens": "{n} answer tokens", "end": "end", "lang": "en",
               "said": "**{a} × {b} = {product}** · {plain} answer tokens with the answer only and {steps} with steps, the end-of-text included; {parts}.",
               "part": "{n} partial product", "parts": "{n} partial products",
               "alt": "The prompt of the chosen pair in grey, then its answer in blue, once with only the product and once with its steps, each with the number of answer tokens."}
    _play = L.explore("formats", "example", lambda _v: C.formats_frame(int(_v), _labels), start=2, values=list(range(1, len(C.PAIRS) + 1)),
                      role="", task="How many answer tokens does each format need at 3 × 3? And how many partial products does a 4 × 4 product need, which the training never shows?")
    _play
    return


@app.cell(hide_code=True)
def text_play_formats_after():
    _a, _b = C.PAIRS[4]
    _n = C.tokens_frame(5, {"title": "", "plain": "", "steps": "", "tokens": "", "end": "", "alt": ""})[1]
    mo.md(rf"""
    At 3 × 3, {_a} × {_b} takes {_n['steps']} answer tokens with steps against {_n['plain']}: about {round(_n['steps'] / _n['plain'])} times as many, in training and in every answer. A 4 × 4 product needs four partial products and three sums; the training shows at most three and two.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(rf"""
    ### Recipe

    1. Draw pairs of numbers of 1, 2 or 3 digits each; keep {C.HELD} pairs of every length out of the training: 2 × 2, 3 × 3, and 4 × 4, longer than any in training.
    2. Write each pair as its prompt and its answer, in one format.
    3. Turn them into ids and labels with `sft_example`, the prompt masked, and into batches with `sft_batch`.
    4. Train the same model on each format with the same steps, batch, learning rate and seed.
    5. Let each model answer the held-out pairs greedily, read its product with a program, and count.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Steps 2 and 3 for 12 × 34: the answer with its steps, then its ids and labels. `C.encode` gives the id of every character, as in [Project: train your GPT on a book](/course/open/gpt-train-chars?lang=en): the characters sorted, character i has id i. Run the cell.
    """)
    return


@app.cell
def example_twelve():
    _a, _b = 12, 34
    # 2. the digits of b from the right, a partial product for each, and the running sum
    _digits = [ord(_c) - ord("0") for _c in reversed(str(_b))]
    _parts = [f"{_a}*{_d}={_a * _d}" for _d in _digits]
    _first = _a * _digits[0]
    _add = _a * _digits[1] * 10 ** 1
    _answer = ",".join(_parts) + ">" + f"{_first}+{_add}={_first + _add}" + ">" + f"{_a * _b}"
    print(_digits, _parts)
    print(_answer)
    # 3. the ids, and a label for every id: -100 for the prompt, then the answer and the end-of-text
    _prompt = C.encode(f"{_a}*{_b}=")
    _ids = _prompt + C.encode(_answer) + [C.END]
    _labels = [-100] * len(_prompt) + C.encode(_answer) + [C.END]
    print(_ids[:10], "...", _ids[-3:], len(_ids))
    print(_labels[:10], "...", _labels[-3:], len(_labels))
    return


@app.cell(hide_code=True)
def text_after_example():
    _answer = C.course_functions()[0](12, 34)
    mo.md(rf"""
    The 3 of 34 stands for 30, so 12 × 3 = 36 counts as 360: 48 + 360 = 408. The prompt gets six labels of -100, the {len(_answer)} tokens of the answer and the end-of-text, id {C.END}, their own ids: {len(_answer) + 7} of each. In the demo, your `sft_example` makes these lists.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_answer():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The answer with its steps

        Complete `steps_answer(a, b)`: step 2 for the format with steps, without the end-of-text. The box starts with the digits of `b` from the right, as numbers: `ord(c) - ord("0")`, as in [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en). The check calls `steps_answer(17, 21)`, which should give `"17*1=17,17*2=34>17+340=357>357"`, and tries one digit, zeros and four digits. Once it is right, it is saved as `llm/steps_answer.py`.
        """),
        L.task("Complete `steps_answer`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_answer():
    box_answer = L.code_box("steps-answer", lines=17)
    box_answer
    return (box_answer,)


@app.cell(hide_code=True)
def check_answer(box_answer):
    result_answer = L.check("steps-answer", box_answer.value)
    result_answer
    return (result_answer,)


@app.cell(hide_code=True)
def hints_answer():
    L.hints(
        "steps-answer",
        ["One partial product per digit: `parts.append(f\"{a}*{d}={a * d}\")`. The digit at place `i` counts `10 ** i` times: write the sum of `total` and `a * digits[i] * 10 ** i`, then add it to `total`."],
        why="Each step is one that a single forward pass can do: a number times one digit, or one sum. The product comes last, after the second `>`, where the checker of exercise 2 reads it.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_read():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A program that reads the product

        Complete `read_product(text)`, the checker of step 5: the part after the last `>`, or the whole text when it has none, if it is all digits (`.isdigit()`), else `None`. The product stays text: the demo compares it with `f"{a * b}"`. The check also reads the empty text and answers that stop halfway. Once it is right, it is saved as `llm/read_product.py`.
        """),
        L.task("Complete `read_product`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_read():
    box_read = L.code_box("read-product", lines=6)
    box_read
    return (box_read,)


@app.cell(hide_code=True)
def check_read(box_read):
    result_read = L.check("read-product", box_read.value)
    result_read
    return (result_read,)


@app.cell(hide_code=True)
def hints_read():
    L.hints(
        "read-product",
        ["`text.split(\">\")[-1]` is the last part, also when the text has no `>`; `\"\".isdigit()` is `False`."],
        why="A model that is still learning writes broken answers. A checker that says `None` for them counts them as wrong, and never as a product.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    _s, _p = C.per_step("cpu", "steps", 1800), C.per_step("cpu", "plain", 1800)
    mo.vstack([
        mo.md(rf"""
        ### Train both models

        The button trains a new ModernGPT of {C.number(C.PARAMS, 0, 'en')} parameters (d = {C.SIZES[1]}, {C.SIZES[4]} blocks, {C.SIZES[2]} query heads, {C.SIZES[3]} key and value heads) on each format in turn: {C.number(C.RUNS['cpu'][0][1], 0, 'en')} steps of {C.BATCH} pairs drawn with seed 0, never a held-out one, with your `steps_answer`, `sft_example`, `sft_batch` and `pretrain_step`; AdamW at {C.number(C.LR, 3, 'en')}, weight decay {C.number(C.WEIGHT_DECAY, 1, 'en')}, with the warmup and warmdown of [Reading an open training recipe](/course/open/llm-open-recipes?lang=en). Then each model answers the {len(C.TESTS) * C.HELD} held-out pairs greedily, and your `read_product` reads every answer.

        On the course's CPU with {C.threads('cpu')} threads both runs took {C.number(C.plan_minutes('cpu'), 0, 'en')} minutes, {C.number(_s, 2, 'en')} s per step with steps and {C.number(_p, 2, 'en')} s without. On a GPU the demo also trains the answer alone for {C.number(C.RUNS['gpu'][2][1], 0, 'en')} steps, four times as many, about the time the run with steps takes on that CPU; the course recorded that run on its CPU too, with {C.threads('gpu')} threads, not on a GPU. A probe of {C.PROBE} steps of each format says first how long it takes on yours; without the time, the course's runs and models stand in.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def probe_button_cell():
    probe_button = mo.ui.run_button(label=f"Time {C.PROBE} steps of each")
    probe_button
    return (probe_button,)


@app.cell(hide_code=True)
def demo_probe(box_answer, probe_button, result_answer):
    _note = None
    if not probe_button.value:
        _note = None
    elif result_answer.state != "correct":
        _note = "*Make `steps_answer` right in exercise 1 first: the probe trains with it.*"
    else:
        mo.output.replace(mo.md("**Timing** …"))
        _t = C.tools()
        _p = "run" if _t == "run" else C.probe(run_code(box_answer.value).names["steps_answer"], _t)
        if isinstance(_p, str):
            _note = "*The probe stops with an error. Do your modules in `work/projects/llm/` still pass their checks? Delete a file you changed, and check its exercise again.*"
        else:
            _long = " That is long: the course's runs below stand in, and its models answer under *Ask both models*. Press *Train both models* when you have the time." if _p["minutes"] > 20 else ""
            _note = f"On this computer a step took {C.number(_p['seconds']['steps'], 2, 'en')} s with steps and {C.number(_p['seconds']['plain'], 2, 'en')} s answer only: all runs of the {_p['kind'].upper()} plan take about **{C.number(_p['minutes'], 0, 'en')} minutes**.{_long}"
    mo.md(_note) if _note else None
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train both models")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_train(box_answer, box_read, result_answer, result_read, train_button):
    trained = None
    if result_answer.state != "correct" or result_read.state != "correct":
        _out = mo.md("*Make `steps_answer` and `read_product` right in exercises 1 and 2 first: the demo trains with the first and counts with the second.*")
    elif not train_button.value:
        _out = mo.md("*Press the button to train both models.*")
    else:
        _plan = C.where()["kind"]
        _names = {"steps": "With steps", "plain": "Answer only"}
        _lives = {}

        def _show(_d):
            _key = (_d.get("format"), _d.get("steps"))
            _name = f"{_names[_key[0]]}, {C.number(_key[1], 0, 'en')} steps" if _key[0] else ""
            if _d["phase"] == "start":
                _live = L.live({"title": f"{_names[_key[0]]}: loss of every step", "x": "step", "y": "loss",
                                "alt": f"The loss of every step so far of your run, {_name.lower()}, in grey, with its mean as the thicker line, and the course's run of the same plan dashed."},
                               _key[1], low=0.0)
                _live.reference(C.live_reference(_plan, *_key), "", role="loss")
                _live.above = [mo.Html(_l.picture()) for _l in _lives.values()]
                _lives[_key] = _live
                _live.line(f"**{_name}** · run {_d['i']} of {_d['n']} …")
            elif _d["phase"] == "train":
                _lives[_key].extend(_d["fresh"], f"**{_name}** · step {C.number(_d['step'], 0, 'en')} of {C.number(_d['steps'], 0, 'en')} · loss {C.number(_d['loss'], 3, 'en')} · {C.number(_d['seconds'] / 60, 1, 'en')} minutes · dashed: the course's run")
            elif _d["phase"] == "measure":
                _lives[_key].line(f"**{_name}** · the model answers the {len(C.TESTS) * C.HELD} held-out pairs …")

        _r = C.run(run_code(box_answer.value).names["steps_answer"], run_code(box_read.value).names["read_product"], _show)
        _charts = [mo.Html(_l.picture()) for _l in _lives.values()]   # they stay after the run
        if isinstance(_r, str):
            _out = mo.vstack(_charts + [mo.md({
                "steps": "*Your `steps_answer` stops with an error on a pair of the training, or writes a character that is no token of the model. Does it still pass its check?*",
                "example": "*Your `sft_example` stops with an error on a pair. Does it still pass its check in [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en)?*",
                "batch": "*Your `sft_batch` stops with an error on a batch of pairs. Does it still pass its check in [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en)?*",
                "read": "*Your `read_product` stops with an error on an answer of the model. Does it still pass its check?*",
                "run": "*The run stops with an error. Do your modules in `work/projects/llm/` still pass their checks? Delete a file you changed, and check its exercise again.*",
                "memory": "*The memory of your device is full. Close other programs, and press the button again.*",
                "diverged": "*The loss is no longer a number: the training ran off. Does your `pretrain_step` use the learning rate it is given?*",
            }[_r])])
        else:
            trained = _r
            _out = mo.vstack(_charts)
    _out
    return (trained,)


@app.cell(hide_code=True)
def demo_result(trained):
    _mine = trained if trained is not None else C.saved()
    _plan = _mine["kind"] if _mine else "cpu"
    _names = {"steps": "with steps", "plain": "answer only"}
    _rows = []
    for _i, (_fmt, _steps) in enumerate(C.RUNS[_plan]):
        _label = f"{_names[_fmt]}, {C.number(_steps, 0, 'en')} steps"
        _course = C.course_runs(_plan, _fmt, _steps)
        if _mine:
            _r = _mine["runs"][_i]
            _rows.append(f"| {_label}, yours | " + " | ".join(f"**{_r['accuracy'][str(_n)]['right']}**" for _n in C.TESTS)
                         + f" | {C.number(_r['seconds'] / _r['steps'], 2, 'en')} | {C.number(C.answer_tokens(_r), 0, 'en')} |")
        _rows.append(f"| {_label}, the course's seeds | " + " | ".join(C.listed([_c['accuracy'][str(_n)]['right'] for _c in _course], 'en') for _n in C.TESTS)
                     + f" | {C.number(C.per_step(_plan, _fmt, _steps), 2, 'en')} | {C.number(C.answer_tokens(_course[0]), 0, 'en')} |")
    _table = "| run | 2 × 2 | 3 × 3 | 4 × 4 | s per step | answer tokens per pair |\n|---|---|---|---|---|---|\n" + "\n".join(_rows)
    _stand = "" if _mine else "\n\n*The course's runs stand in until you train your own.*"
    _verdict = ""
    if _mine:
        _r = _mine["runs"][0]
        _ok = all(_r["accuracy"][str(_n)]["right"] >= C.THRESHOLD[_plan][str(_n)] for _n in ("2", "3"))
        _verdict = (f"\n\nYour run with steps started at a loss of {C.number(_r['first'], 1, 'en')}, near ln {C.SIZES[0]} = {C.number(C.START_LOSS, 1, 'en')}, as a new model should. "
                    f"The check asks at least {C.THRESHOLD[_plan]['2']} of {C.HELD} right at 2 × 2 and {C.THRESHOLD[_plan]['3']} at 3 × 3: the lowest of the course's three seeds, {min(C.MEASURED[_plan]['steps-1800']['2'])} and {min(C.MEASURED[_plan]['steps-1800']['3'])}, less a margin for another computer. "
                    + ("Yours got them: your `steps_answer` and the modules of your project train as they should." if _ok else
                       "Yours did not. Do your modules in `work/projects/llm/` still pass their checks? Then compare your `steps_answer` with the solution under exercise 1."))
    mo.md(f"Right of {C.HELD} held-out pairs at each length, with seconds a training step and tokens an answer takes, its end-of-text included:\n\n{_table}{_verdict}{_stand}")
    return


@app.cell(hide_code=True)
def text_after_demo():
    _s, _p = K["runs"][0], K["runs"][1]
    mo.md(rf"""
    **Reading the runs.** With steps the loss ends far lower, {C.number(_s['losses'][-1]['loss'], 3, 'en')} against {C.number(_p['losses'][-1]['loss'], 2, 'en')} in the course's runs of seed 0, but that says little: most tokens of an answer with steps follow from the ones before, so their loss is low even when the product is wrong. Compare what each model gets right. And the steps cost: about {C.number(_s['tokens'] / _p['tokens'], 0, 'en')} times the tokens to train on for as many pairs, and about {C.number(C.answer_tokens(_s) / C.answer_tokens(_p), 0, 'en')} times the tokens for every answer, test-time compute the answer alone never spends.
    """)
    return


@app.cell(hide_code=True)
def text_replay():
    mo.md(rf"""
    ### Your run, step by step

    At the start and after every fifth of each run, the demo lets both models answer the first {C.SNAP} held-out 3 × 3 pairs. Go through those moments: both losses and the right answers so far, solid with steps and dashed answer only, and what each model wrote for {C.HELD_OUT[3][0][0]} × {C.HELD_OUT[3][0][1]}.
    """)
    return


@app.cell(hide_code=True)
def replay_cell(trained):
    _info = trained if trained is not None else C.saved()
    _run = _info if _info is not None else K
    _whose = "your runs" if _info is not None else "the course's runs"
    _labels = {"title_loss": "The loss along both runs", "title_right": f"Right of {C.SNAP} 3 × 3 pairs", "x": "step",
               "y_loss": "loss", "y_right": "right", "lang": "en",
               "alt": f"Left, the loss of {_whose}, solid with steps and dashed answer only, with a dashed line at the chosen moment. Right, how many of {C.SNAP} held-out 3 × 3 pairs each model got right at every moment up to then, solid with steps and dashed answer only.",
               "start": "Before the first step", "after": "After {step} steps", "right": "right of {n}: with steps {steps}, answer only {plain}",
               "steps": "With steps", "plain": "Answer only", "longer": "And three of its answers with steps to 4 × 4 pairs:"}
    _steps = [_s["step"] for _s in C.pair_of(_run)[0]["snapshots"]]
    _block = L.explore("replay", "step", lambda _v: C.replay_frame(_run, _steps.index(int(_v)), _labels), start=_steps[-1], values=_steps, role="",
                       task=f"At which moment does the model with steps first get products right? Which part of its answer is right before the product is?")
    mo.vstack([_block] if _info is not None else [mo.md("*Until you train your own models, you go through the course's runs of seed 0 here.*"), _block])
    return


@app.cell(hide_code=True)
def text_replay_after():
    _s, _p = K["runs"][0]["snapshots"], K["runs"][1]["snapshots"]
    _first = next(_x for _x in _s if _x["right"] > 0)
    mo.md(rf"""
    In the course's run of seed 0 (your own may differ a little), the model with steps gets its first products right after {C.number(_first['step'], 0, 'en')} steps, {_first['right']} of {C.SNAP}, and {_s[-1]['right']} at the end. It writes the form of an answer from the first moment after the start, with wrong numbers: {C.shown(_s[1]['text'])}. After {C.number(_s[3]['step'], 0, 'en')} steps its running sums are right, while 169 × 6 = {C.first_part(_s[3]['text'])} is still wrong, and so the product is; after {C.number(_s[4]['step'], 0, 'en')} the whole answer is right. The model on the answer alone gets none of them right at any moment: after the start it writes a number of about the right length, {C.shown(_p[-1]['text'])} at the end.
    """)
    return


@app.cell(hide_code=True)
def text_ask():
    mo.md(r"""
    ### Ask both models

    Type two numbers of up to 4 digits, such as `169*186`. Both models answer greedily: yours once you have trained them, else the course's.
    """)
    return


@app.cell(hide_code=True)
def ask_cell():
    ask_box = mo.ui.text(value="169*186", label="Your pair")
    ask_button = mo.ui.run_button(label="Ask both models")
    mo.hstack([ask_box, ask_button], align="end", justify="start")
    return ask_box, ask_button


@app.cell(hide_code=True)
def demo_ask(ask_box, ask_button, box_read, result_read, trained):
    _note = None
    if ask_button.value:
        _pair = C.asked(ask_box.value)
        if _pair is None:
            _note = "*Type two numbers of 1 to 4 digits with a `*` between them, such as `169*186`.*"
        else:
            _read = run_code(box_read.value).names["read_product"] if result_read.state == "correct" else C.course_functions()[1]
            _out = C.ask(_pair, _read, C.course_functions()[0])
            _whose = "your models" if _out[0][4] == "yours" else "the course's models"
            _lines = [f"- {'With steps' if _f == 'steps' else 'Answer only'}: {C.shown(_text)} → " + ("no product" if _got is None else f"{_got}, " + ("right" if _ok else "wrong"))
                      for _f, _text, _got, _ok, _ in _out]
            _note = f"{_pair[0]} × {_pair[1]} = {C.whole(_pair[0] * _pair[1], 'en')}, by {_whose}:\n\n" + "\n".join(_lines)
    mo.md(_note) if _note else None
    return


@app.cell(hide_code=True)
def text_exercise_longer():
    mo.md(r"""
    ### Exercise 3. Longer numbers
    """)
    return


@app.cell(hide_code=True)
def explain_longer():
    explain_longer_cell = L.explain(
        "longer",
        question=f"The model with steps gets most 3 × 3 products right and none of {C.HELD} 4 × 4 ones. Look at what it wrote for 4 × 4 at the last moment of the replay. Why does it fail, when every kind of step it writes is one it learned?",
        model_answer="It never saw four partial products, nor a number of four digits: it mostly writes the plan of a 3-digit product, three partial products and their sums, and now and then loses the form altogether. It learned the form of its examples, not a method for any length, and the places of a longer answer were never trained. Small transformers trained on arithmetic rarely handle longer numbers than in training (Lee and others, 2023); longer training numbers, or a way to mark where each digit stands, help, but not robustly (McLeish and others, 2024; Zhou and others, 2024).",
        points=[
            "the training set the length: it never showed four partial products or a 4-digit number",
            "the model writes the form it learned, three partial products, not a general method",
            "this is length generalization, and it is an open problem for small transformers",
        ],
    )
    explain_longer_cell
    return (explain_longer_cell,)


@app.cell(hide_code=True)
def text_exercise_fair():
    mo.md(r"""
    ### Exercise 4. Is the comparison fair?
    """)
    return


@app.cell(hide_code=True)
def explain_fair():
    _long = C.MEASURED["gpu"]["plain-7400"]["3"]
    explain_fair_cell = L.explain(
        "fair",
        question=f"With the same {C.number(C.RUNS['cpu'][0][1], 0, 'en')} steps, the model on the answer alone saw as many pairs, but far fewer tokens. In the course's runs of the GPU plan it trained four times as many steps, on about as many tokens as the run with steps, and still got {C.listed(_long, 'en')} of {C.HELD} 3 × 3 products right. Is the comparison fair? What would you add to be sure?",
        model_answer="It is fair in the model, the data, the steps and the seed, and four times the steps, on about as many tokens, show that the extra tokens are not the whole story. It is not fair in tuning effort: one learning rate for both, the one the research for this lesson took for the format with steps. I would tune it for each format, run more seeds, and try a larger model or a longer run: GPT-2 Small reached 4 × 4 without steps (Deng and others, 2024). And I would report the cost: more training time per step and many more tokens per answer.",
        points=[
            "one thing changed, the format; equal steps and four times the steps both checked",
            "the learning rate was not tuned for the answer alone: tuning effort",
            "more seeds, and a larger model or a longer run",
            "the steps cost training time and answer time",
        ],
    )
    explain_fair_cell
    return (explain_fair_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(rf"""
    ## Wrap-up

    **How this connects.** [How an LLM reasons](/course/open/llm-reasoning-idea?lang=en) prompted a large model for its steps; here a model learned them from its data, which is how chains of thought get into a model in the first place. Your project `llm` has `steps_answer.py` and `read_product.py` now, and after the demo both models in `arith_steps.pt` and `arith_plain.pt`, with their runs in `arith.json`.

    **And reinforcement learning?** The next step would be [Reinforcement learning with verifiable rewards](/course/open/llm-rlvr?lang=en), with your `read_product` as the reward. The course tried it while researching this lesson, on a model with steps that got {C.number(C.RL['before'], 1, 'en')}% of 3 × 3 right, and {C.RL['steps']} steps of GRPO did not help: {C.number(C.RL['1e-4'], 1, 'en')}% after, at a learning rate of 0.0001; {C.number(C.RL['3e-4'], 1, 'en')}% at 0.0003; and 0% at 0.001. Yet pass@8 of {C.RL['pass8']}%, against {C.number(C.RL['one'], 1, 'en')}% for one sample, says there was room. Making it work is open; the course adds the step once it does.

    **Going further.** Vote over sampled answers of your model with steps, as in *How an LLM reasons*. Train both formats on addition, where a small model gets far without steps. Or check every step `x*y=z` and `x+y=z` of an answer, not only its product, and look for right products with a wrong step.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_answer, result_read, explain_longer_cell, explain_fair_cell):
    L.wrap_up(result_answer, result_read, explain_longer_cell, explain_fair_cell)
    return


@app.cell(hide_code=True)
def look_back():
    _s = M["steps-1800"]["3"]
    _look = L.choice(
        "look-seeds",
        question=f"### Looking back\n\nThe course ran each format with three seeds. With steps they got {C.listed(_s, 'en')} of {C.HELD} 3 × 3 products right. What do three seeds tell you that one run would not?",
        options={
            "right": "How much the result changes with only the seed, so that you can see the difference between the formats is far bigger",
            "exact": "The exact accuracy of the method, as the mean of the three",
            "nothing": "Nothing: the same code gives the same number with any seed",
        },
        correct="right",
        why={
            "right": "Another seed gives other starting weights and other pairs, and another number. A difference between the formats far bigger than that spread is no accident.",
            "exact": "Three runs give a mean and a spread, not an exact value. See [One run is not a result: seeds and variance](/course/open/res-seeds-variance?lang=en).",
            "nothing": "The seed sets the starting weights and the order of the pairs: the three runs differ. See [One run is not a result: seeds and variance](/course/open/res-seeds-variance?lang=en).",
        },
    )
    _look
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
