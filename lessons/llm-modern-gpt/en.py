# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-modern-gpt. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("llm-modern-gpt", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Grouped-query attention](/course/open/llm-gqa?lang=en) and [The KV cache](/course/open/llm-kv-cache?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_group():
    quick_group = L.choice(
        "quick-group",
        question="A layer has 4 query heads and 2 key and value heads. Which key head does query head 3 use? Heads count from 0.",
        options={
            "right": "Key head 1: query heads 2 and 3 share it",
            "own": "Key head 3: every query head has its own",
            "first": "Key head 0: all query heads share the first",
        },
        correct="right",
        why={
            "own": "That is multi-head attention, with as many key heads as query heads. With 2 key heads, each serves a group of 4 / 2 = 2 query heads: see [Grouped-query attention](/course/open/llm-gqa?lang=en).",
            "first": "That is multi-query attention, with one key head. With 2, query heads 0 and 1 share key head 0, and 2 and 3 share key head 1: see [Grouped-query attention](/course/open/llm-gqa?lang=en).",
            "right": "`int(3 / (4 / 2))` = 1. Your ModernAttention shares its key and value heads in the same way.",
        },
    )
    quick_group
    return (quick_group,)


@app.cell(hide_code=True)
def quick_check_cache():
    quick_cache = L.choice(
        "quick-cache",
        question="Your GPT writes with a KV cache and has written 50 tokens. What goes through its layers at the next step?",
        options={
            "all": "All 50 tokens: the cache makes each of them faster",
            "none": "Nothing at all: the cache already holds the next token and its logits",
            "right": "Only the newest token: the cache has the others' keys and values",
        },
        correct="right",
        why={
            "all": "Then the cache would save nothing. The keys and values of the 50 tokens are kept, so only the newest one is projected and goes through the MLPs: see [The KV cache](/course/open/llm-kv-cache?lang=en).",
            "none": "The cache holds keys and values, not tokens. The next token still comes from the logits of a step: see [The KV cache](/course/open/llm-kv-cache?lang=en).",
            "right": "Its query reads all keys and values in the cache. In exercise 4 your own GPT writes this way.",
        },
    )
    quick_cache
    return (quick_cache,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(f"""
        ## Why this matters

        Your GPT of [Project: your GPT on your own tokens](/course/open/gpt-train-bpe?lang=en) is close to GPT-2 of 2019: layer normalization, an MLP (with ReLU, where GPT-2 has GELU), a table of learned places at the input, and as many key and value heads as query heads. Llama, Mistral and Qwen changed these four parts, and they write with a KV cache. You built each of them on its own in the last five lessons. In this project they go into your GPT, and you measure what each one does to the loss: the same stories, the same tokens and the same number of steps as your GPT on tokens, so that only the change differs.

        **An everyday example.** A baker who changes four things in a recipe at once and gets a better bread does not know which change did it. Change one thing per bake, bake each version a few times, and taste.
        """),
        draw.project_steps(["Count letters and pairs", "A bigram model by counting", "The same model, learned by a neural network",
                            "Your own tokenizer", "Your own GPT", "Pretrain a small LLM", "Teach it to follow instructions and preferences"],
                           4, here_label="← today", title="Where your project stands: your own LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_change():
    _best = {_k: C.mean_best(_k) for _k in C.CHANGES}
    _step = {_k: _best[_k] - _best[C.CHANGES[_i - 1]] for _i, _k in enumerate(C.CHANGES) if _i > 0}
    _top = min(_step, key=_step.get)
    _label = {"rms": "RMSNorm", "swiglu": "SwiGLU", "rope": "RoPE", "gqa": "Grouped-query attention"}
    guess_change = L.guess(
        "change",
        question=f"The course trained the GPT on tokens of [Project: your GPT on your own tokens](/course/open/gpt-train-bpe?lang=en) with the CPU plan, then added the changes one by one: RMSNorm, then SwiGLU, then RoPE, then grouped-query attention, each version three times. The GPT on tokens ended at {C.number(_best['gpt'], 2, L.lang)} per token. Which change do you think lowered the validation loss the most?",
        options={
            "rms": "RMSNorm in place of layer normalization",
            "swiglu": "SwiGLU in place of the MLP with ReLU",
            "rope": "RoPE in place of the table of places",
            "gqa": "Grouped-query attention, 2 key and value heads instead of 4",
        },
        reveal=f"**{_label[_top]}: {C.signed(_step[_top], 2, L.lang)} per token.** " + "; ".join(f"{_label[_k]}: {C.signed(_step[_k], 2, L.lang)}" for _k in _step) + ", each the change in the average of three runs. The demo shows every run, next to the spread between seeds.",
    )
    guess_change
    return (guess_change,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Four parts change, the rest stays.** The embedding of the tokens, the residual stream, pre-norm, the output head and the training run stay as in your GPT. Four parts change:

        | In your GPT | In the ModernGPT |
        |---|---|
        | layer normalization | RMSNorm: no mean, no bias terms |
        | MLP from $d$ to $4d$ to $d$, with ReLU | SwiGLU with a hidden size of `int(8 * d / 3)`: three weight matrices instead of two, so with about as many parameters |
        | a table of learned places, added at the input | RoPE in every attention, on the queries and keys; no table, so a ModernGPT has no window size among its sizes |
        | as many key and value heads as query heads | grouped-query attention: `kv_heads` key and value heads, each shared by `h / kv_heads` query heads |

        **RoPE on every head on its own.** Your `rope` turns the pairs of a vector of $d$ numbers, each pair at its own speed. In attention, a vector is one head: the `d_head` numbers of that head. So cut the queries into heads, turn each head with `rope`, and put them side by side again with `torch.cat`; the same for the keys, with `kv_heads` heads.

        **The KV cache inside the GPT.** While writing, every attention gets a cache of its own: a dict. The first call, with the whole start, fills it with the keys and values of those tokens. Every next call gets one token. That token stands at place $t$, the number of keys already in the cache, so RoPE must turn it as a token at place $t$, not 0: your `rope` gets an argument `start`, the place of its first row. Then the keys and values of the cache come first, and all of them stay in the cache. With one query, your `gqa` makes a mask of 1 by 1 that hides nothing, so the new token reads every key: all of them are earlier tokens, or itself.

        **Measure every change on its own.** The course added the changes one at a time, each on top of the ones before, and trained every version three times, with seeds 0, 1 and 2, on the stories, tokens and steps of your GPT on tokens. Even with the same seed, a version with a new part draws other starting values and its mini-batches in another order, so two runs already differ a little; a change counts only when it moves the loss by more than that spread.
        """),
        L.watch_out("RoPE turns pairs of numbers, so a head needs an even number of them: `d / h` must be even. The plans of this lesson have heads of 32 and 64 numbers."),
        mo.md(r"""
        The picture: the ModernGPT's block, with what each part replaces under it, in grey. The residual stream runs straight down, as in your Block.
        """),
        mo.Html(C.block_picture({"title": "One block of the ModernGPT", "input": "x: the token embeddings, without places", "input_was": "was: tokens plus the table of places",
                                 "output": "to the next block", "norm": "RMSNorm", "norm_was": "was: layer normalization",
                                 "attention": "grouped-query attention, RoPE", "attention_was": "was: h key and value heads",
                                 "mlp": "SwiGLU: d → 8d/3 → d", "mlp_was": "was: d → 4d → d, with ReLU", "stream": "residual stream",
                                 "alt": "The residual stream runs from x, the token embeddings without places, straight down to the next block. Two branches leave it: RMSNorm, then grouped-query attention with RoPE, added back to the stream; then RMSNorm, then SwiGLU from d to 8d/3 to d numbers, added back again. Under every part it says what it replaces: layer normalization, attention with h key and value heads, and an MLP from d to 4d to d with ReLU; at the input, the table of places."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To make your GPT a modern one, and to check each change:

    1. Put the four parts in your project, and give `rope` a place to start from: `start`.
    2. Attention: queries for `h` heads, keys and values for `kv_heads` heads, every head of the queries and keys turned by RoPE from `start`; with a cache, its keys and values first, and keep them all.
    3. Block and GPT: your Block and GPT with RMSNorm and SwiGLU, without the table of places; every block gets its own cache.
    4. Train it on the tokens of your GPT on tokens, with the same plan and seed, and compare its loss with the course's runs, change by change.
    5. Write with the caches: the start once, then one token per step.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Steps 2 and 3 in numbers, for the CPU plan: what each change does to the parameters, and what grouped-query attention does to the KV cache of a full window.
    """)
    return


@app.cell
def example_counts():
    _d, _h, _kv_heads, _n_layers, _block = 128, 4, 2, 5, 128
    _d_head = int(_d / _h)
    # 2. keys and values: for h heads, and for kv_heads heads
    _kv = 2 * (_d * _d)
    _kv_gqa = 2 * (_d * _kv_heads * _d_head)
    print("keys and values:", _kv, "->", _kv_gqa, "per block")
    # 3. the MLP of your Block, d -> 4d -> d with bias terms, and SwiGLU with a hidden size of int(8 * d / 3)
    _hidden = int(8 * _d / 3)
    _mlp = _d * 4 * _d + 4 * _d + 4 * _d * _d + _d
    _swiglu = 3 * _d * _hidden
    print("MLP:", _mlp, "-> SwiGLU:", _swiglu, "with hidden size", _hidden)
    # 3. the normalizations, two per block and one at the end: RMSNorm has no bias terms
    _norms = 2 * _n_layers + 1
    print("normalizations:", _norms * 2 * _d, "->", _norms * _d)
    # 3. the table of places: block rows of d numbers, gone
    print("table of places:", _block * _d, "-> 0")
    # the KV cache of a full window: every block keeps a key and a value per token, for every key and value head
    _cache = _n_layers * 2 * _h * _d_head * _block
    _cache_gqa = _n_layers * 2 * _kv_heads * _d_head * _block
    print("KV cache:", _cache, "->", _cache_gqa, "numbers")
    return


@app.cell(hide_code=True)
def text_after_example():
    _p = C.plan_for("cpu")
    _hidden = int(8 * 128 / 3)
    mo.md(f"""
    Grouped-query attention halves the keys and values, in the parameters and in the cache. SwiGLU with a hidden size of {_hidden} has {C.grouped(3 * 128 * _hidden, L.lang)} parameters where the MLP had {C.grouped(2 * 128 * 512 + 512 + 128, L.lang)}: nearly the same, so a lower loss cannot come from more parameters. All together, the ModernGPT of the CPU plan has {C.grouped(_p['params'], L.lang)} parameters, your GPT on tokens {C.grouped(_p['params_gpt'], L.lang)}.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_parts():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Your parts in your project

        Step 1 of the recipe. Copy your `RMSNorm` of [RMSNorm](/course/open/llm-rmsnorm?lang=en), your `SwiGLU` of [Modern MLP blocks: GELU, SiLU and SwiGLU](/course/open/llm-swiglu?lang=en), your `rope` of [Rotary position embeddings (RoPE)](/course/open/llm-rope?lang=en) and your `gqa` of [Grouped-query attention](/course/open/llm-gqa?lang=en) into the box: open each lesson and copy your code from its box. Then give `rope` one more argument, `start=0`: the place of its first row. The check runs every part as its own lesson did, and also `rope(x, start=3)`: two rows that stand at places 3 and 4.
        """),
        L.task("Put your four parts in the box, and let the places of `rope` run from `start`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_parts():
    box_parts = L.code_box("parts", lines=40)
    box_parts
    return (box_parts,)


@app.cell(hide_code=True)
def check_parts(box_parts):
    result_parts = L.check("parts", box_parts.value)
    result_parts
    return (result_parts,)


@app.cell(hide_code=True)
def hints_parts():
    L.hints(
        "parts",
        [
            "The four parts stay as they were; only `rope` changes, in two places: its `def` line and its angles.",
            "Give `start` the value 0 on the `def` line for when it is not given, so that `rope(x)` works as before. In the angles, every place moves up by `start`: add it to `torch.arange(n)` inside brackets, before `[:, None]`.",
        ],
        why="A row of `rope` turns by its place times `theta`. With `start`, the rows of a piece of text turn as they would in the whole text, from place `start` on: so a new token in exercise 2 is turned at its own place, and its scores with the earlier keys are those of the whole text.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_attention():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. ModernAttention

        Step 2 of the recipe. Complete `ModernAttention(d, h, kv_heads)`, with your `rope` and `gqa` from exercise 1. `turn(x, n_heads, start)` turns every head of `x` on its own. `forward(x, cache=None)` works without a cache while training; while writing, `cache` is a dict that keeps the keys and values of the earlier tokens, under `"k"` and `"v"`. The check makes small ones, such as `ModernAttention(8, 4, 2)`, and looks at their weights, at `turn`, and at a text of 5 tokens: 4 with an empty cache, then the last one on its own, which must give what the whole text gives at its place.
        """),
        L.task("Make `ModernAttention` turn the queries and keys of every head, and keep its keys and values in the cache."),
    ])
    return


@app.cell(hide_code=True)
def exercise_attention():
    box_attention = L.code_box("modern-attention", lines=34)
    box_attention
    return (box_attention,)


@app.cell(hide_code=True)
def check_attention(box_attention, result_parts):
    result_parts  # check again when exercise 1 saves your parts
    result_attention = L.check("modern-attention", box_attention.value)
    result_attention
    return (result_attention,)


@app.cell(hide_code=True)
def hints_attention():
    L.hints(
        "modern-attention",
        [
            "`__init__`: the queries `d` numbers wide, the keys and values `kv_heads * self.d_head`, all three without bias terms, and `out` as in your CausalSelfAttention. `turn`: a loop over the `n_heads` heads, `rope` with `start=start` on the columns of each head, and the turned heads side by side with `torch.cat` on the last axis.",
            "`forward`: `turn` the queries with `self.h` heads and the keys with `self.kv_heads`; the values stay as they come. With keys in the cache, put those before the new ones, with `torch.cat` on the axis of the tokens; with a cache, also an empty one, keep all keys and values in it, under `\"k\"` and `\"v\"`.",
        ],
        why="Only the queries and keys are turned, every head as its own vector, so that a score depends on how far apart two tokens are. The cache keeps the keys turned at their own places; a new token is turned at place `start`, after them, and its query reads them all.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_gpt():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. ModernBlock and ModernGPT

        Step 3 of the recipe. Complete `ModernBlock(d, h, kv_heads)` and `ModernGPT(vocab, d, h, kv_heads, n_layers)`: your Block of [The transformer block](/course/open/attn-transformer-block?lang=en) and your GPT of [Assemble your GPT](/course/open/gpt-assemble?lang=en), with the names they had, `mlp` for the SwiGLU, and without the table of places. `forward(ids, caches=None)` gets a list of caches while writing, one dict per block. The check makes small ones, such as `ModernGPT(20, 16, 4, 2, 3)`, counts their parameters, and gives a text of 5 tokens once whole and once as 4 and then 1, with caches.
        """),
        L.task("Make `ModernBlock` and `ModernGPT`, and give every block its own cache."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gpt():
    box_gpt = L.code_box("modern-gpt", lines=36)
    box_gpt
    return (box_gpt,)


@app.cell(hide_code=True)
def check_gpt(box_gpt, result_parts, result_attention):
    result_parts, result_attention  # check again when exercise 1 or 2 saves your module
    result_gpt = L.check("modern-gpt", box_gpt.value)
    result_gpt
    return (result_gpt,)


@app.cell(hide_code=True)
def hints_gpt():
    L.hints(
        "modern-gpt",
        [
            "A block: your Block, with your `RMSNorm(d)` where it had `nn.LayerNorm(d)`, `ModernAttention(d, h, kv_heads)` as `attention`, and your `SwiGLU` with a hidden size of `int(8 * d / 3)` as `mlp`, where it had `lin1` and `lin2`. Its `forward` is that of your Block, with `cache` passed on to the attention.",
            "The GPT: your GPT without `pos`, with `RMSNorm` as `norm`. In `forward`, go through the blocks by their number `i`, so that block `i` can get `caches[i]`; when `caches` is None, call each block without a cache.",
        ],
        why="The block is your Block with RMSNorm and SwiGLU; the attention brings the places, so the GPT starts from the tokens alone. Every block has keys and values of its own, so every block needs a cache of its own.",
    )
    return


@app.cell(hide_code=True)
def text_demo(result_gpt):
    result_gpt  # the note changes when exercise 3 saves your module
    _p = C.plan()
    _d, _h, _layers, _block = _p["sizes"]
    if _p["device"] == "cpu":
        _where = "This computer has no GPU that fits the bigger GPT" + (f" ({_p['small']} has too little memory)" if _p["small"] else "") + ", so the CPU plan runs"
        _time = f" On the course's computer, a CPU with 4 cores, the run took about {round(C.minutes('gqa'))} minutes, on one thread."
    else:
        _where = f"Your computer has a GPU, {_p['name']}, so your ModernGPT trains on it"
        _time = ""
    mo.vstack([
        mo.md(f"""
        ### What your project can do now

        The button below trains your ModernGPT as [Project: your GPT on your own tokens](/course/open/gpt-train-bpe?lang=en) trains your GPT: your tokenizer on the first 90% of the stories, all of them encoded, and your `gpt_loaders` and `Trainer`, with the same plan and seed. {_where}: a ModernGPT of {C.grouped(_p['params'], L.lang)} parameters, with $d = {_d}$, {_h} query heads and {_p['kv_heads']} key and value heads, {_layers} blocks and windows of {_block} tokens, {_p['epochs']} epochs of mini-batches of {_p['batch']}. Then it writes a story that starts with "{C.B.START}".{_time} The stop button of the cell ends the run early; the best parameters so far stay saved, in `work/projects/llm/gpt_modern.pt`.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train your ModernGPT")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_train(train_button, result_gpt):
    result_gpt  # run again when exercise 3 saves your module
    _labels = {"title": "Your run", "x": "epoch", "y": "loss per token", "lang": L.lang, "guess": "guessing evenly", "train": "training", "val": "validation"}
    _names = {"gpt": "your GPT on tokens", "rms": "+ RMSNorm", "swiglu": "+ SwiGLU", "rope": "+ RoPE, no table of places", "gqa": "+ grouped-query attention"}

    def _progress(_p):
        if _p["phase"] == "tokenizer":
            mo.output.replace(mo.md("**Training your tokenizer** on the first 90% of the stories…"))
            return
        if _p["phase"] == "encode":
            mo.output.replace(mo.md("**Encoding the stories** with your `encode_fast`…"))
            return
        _m = 1 if _p["left"] is None else max(1, round(_p["left"] / 60))
        _left = "" if _p["left"] is None else f" · about {_m} {'minutes' if _m > 1 else 'minute'} left"
        _parts = [mo.md(f"**Epoch {_p['epoch']} of {_p['epochs']}** · step {_p['step']} of {_p['steps']}{_left}")]
        if _p["history"]:
            _parts.append(mo.Html(C.history_picture(_p["history"], {**_labels, "alt": "The training and validation loss of your run so far, after every epoch."}, _p["epochs"])))
        mo.output.replace(mo.vstack(_parts))

    _best = {_k: C.mean_best(_k) for _k in C.CHANGES}
    _step = {_k: _best[_k] - _best[C.CHANGES[_i - 1]] for _i, _k in enumerate(C.CHANGES) if _i > 0}
    _short = {"gpt": "GPT on tokens", "rms": "+ RMSNorm", "swiglu": "+ SwiGLU", "rope": "+ RoPE", "gqa": "+ GQA"}
    _course = mo.vstack([
        mo.Html(C.changes_picture([(_names[_k], [_r["best"] for _r in C.runs(_k)]) for _k in C.CHANGES],
                                  {"title": "The course's runs, change by change", "x": "best validation loss per token, CPU plan", "lang": L.lang,
                                   "alt": "One row per GPT, each with the changes of the rows above it, and a dot for each of its three runs: " + "; ".join(f"{_names[_k]}: " + ", ".join(C.number(_r['best'], 2, L.lang) for _r in C.runs(_k)) for _k in C.CHANGES) + "."})),
        mo.md("The best validation loss per token on the CPU plan: the average of the three runs of every version, and the spread, the highest minus the lowest.\n\n| | parameters | average | spread |\n|---|---|---|---|\n" + "\n".join(
            f"| {_short[_k]} | {C.grouped(C.runs(_k)[0].get('params', C.plan_for('cpu')['params_gpt']), L.lang)} | {C.number(C.mean_best(_k), 3, L.lang)} | {C.number(C.spread(_k), 3, L.lang)} |" for _k in C.CHANGES)),
        mo.md(f"""RoPE lowers the loss by far the most, by {C.number(-_step['rope'], 2, L.lang)}: a table of places has to be learned before attention can use it, and RoPE gives every attention the distance between two tokens from the first step on. SwiGLU lowers it by {C.number(-_step['swiglu'], 2, L.lang)}. RMSNorm leaves it as it was, {C.signed(_step['rms'], 3, L.lang)}, within the spread. Grouped-query attention raises it a little, by {C.number(_step['gqa'], 3, L.lang)}: more than the spread, but small next to what it saves, half of the KV cache. The time goes up too: alone on one thread, a run of the GPT on tokens takes about {round(C.minutes('gpt'))} minutes, one of the ModernGPT {round(C.minutes('gqa'))}. Your RMSNorm is a few small steps of PyTorch where `nn.LayerNorm` is one, written in C++, and `turn` turns every head on its own, in a loop."""),
        mo.md(f"""On the GPU plan, with $d = {C.plan_for('gpu')['sizes'][0]}$, {C.plan_for('gpu')['sizes'][2]} blocks and windows of {C.plan_for('gpu')['sizes'][3]} tokens, {C.grouped(C.plan_for('gpu')['params'], L.lang)} parameters on the same stories, the course's runs tell another story. Its ModernGPT reached its lowest validation loss, {C.number(C.runs('gqa', 'gpu')[0]['best'], 2, L.lang)}, after epoch {C.runs('gqa', 'gpu')[0]['best_epoch']}, and then overfit; the GPT on tokens reached {C.number(C.runs('gpt', 'gpu')[0]['best'], 2, L.lang)} after epoch {C.runs('gpt', 'gpu')[0]['best_epoch']}. The ModernGPT learns faster, so with this many parameters for these stories it also starts to learn them by heart sooner. A change that helps in one setting can hurt in another: measure it in the setting you train in."""),
    ])
    if train_button.value:
        _result = C.train(_progress)
    else:
        _result = C.last_run()
    _ready = result_gpt.state == "correct"
    if _result is None:
        if _ready and not train_button.value:
            _out = mo.vstack([mo.md("*Press the button to start the run. Until then, the course's runs:*"), _course])
        else:
            _out = mo.vstack([mo.md("*Finish exercises 1 to 3 first, then press the button: the run uses your ModernGPT. Until then, the course's runs:*"), _course])
        if C.trained()["source"] == "course":
            _c, _b = C.trained(), C.course_bpe()
            _out = mo.vstack([_out, mo.md(f"""And the course's own ModernGPT: the ModernGPT of the CPU plan, trained by the course on {C.grouped(_c['chars'], L.lang)} characters of stories from Cosmopedia, whose license lets the course share a model trained on them, and not on TinyStories. On these stories, harder than TinyStories, its validation loss is {C.number(_c['best'], 2, L.lang)} per token; the course's GPT on tokens of [Project: your GPT on your own tokens](/course/open/gpt-train-bpe?lang=en), on the same stories, ended at {C.number(_b['best'], 2, L.lang)}. It wrote, with temperature 0.8:

> {_c['story'].replace(chr(10), chr(10) + '> ')}""")])
    elif isinstance(_result, str):
        _out = mo.md({
            "run": "*The run stops with an error. Do your modules in `work/projects/llm/` and `work/projects/nn/` still pass their checks? Delete a file you changed, and check its exercise again.*",
            "memory": "*Your GPU ran out of memory. Close other programs that use it, such as a game or another notebook, and press the button again.*",
            "stopped": "*You stopped the run during its first epoch, so nothing of it was saved. Press the button to start again.*",
        }[_result])
    else:
        _h = _result["history"]
        _plan = "gpu" if _result["device"] != "cpu" else "cpu"
        _on = {"cpu": "the CPU", "mps": "the Apple GPU"}.get(_result["device"], _result["name"])
        _limit = C.number(C.THRESHOLD[_plan], 2, L.lang)
        if _result["stopped"]:
            _verdict = f"You stopped the run after epoch {len(_h)} of {_result['epochs']}; the course's runs of the ModernGPT, with all {_result['epochs']} epochs, ended below {_limit} per token."
        elif _result["best"] <= C.THRESHOLD[_plan]:
            _verdict = f"The course's runs of the ModernGPT ended below {_limit} per token, and so did yours: your four changes work as they should."
        else:
            _verdict = f"The course's runs of the ModernGPT ended below {_limit} per token, and yours did not. Check exercises 1 to 3 again, and those of the lessons your tokenizer and Trainer come from."
        _old = C.B.last_run()
        if _old is not None and _old["device"] == _result["device"] and _old["epochs"] == _result["epochs"] and not _old["stopped"] and not _result["stopped"]:
            _diff = _old["best"] - _result["best"]
            _versus = f" Your GPT on tokens of the same plan ended at {C.number(_old['best'], 2, L.lang)} per token: " + (f"your ModernGPT ended {C.number(abs(_diff), 2, L.lang)} {'lower' if _diff > 0 else 'higher'}." if abs(_diff) >= 0.005 else "your ModernGPT ended at about the same.")
        else:
            _versus = f" The course's GPT on tokens of the {_plan.upper()} plan ended at {C.number(C.mean_best('gpt', _plan), 2, L.lang)} per token."
        _saved = "" if not _result.get("kept") else " Your earlier run got a lower validation loss, so `work/projects/llm/gpt_modern.pt` keeps the parameters of that run."
        _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{_m}`" for _m in _result["stand_ins"]) + "."
        _story = "" if not _result.get("story") else "\n\nThe story your ModernGPT wrote, with temperature 0.8:\n\n> " + _result["story"].replace("\n", "\n> ")
        _out = mo.vstack([
            mo.Html(C.history_picture(_h, {**_labels, "alt": f"The training and validation loss of your run after every epoch: the validation loss is lowest after epoch {_result['best_epoch']}, at {C.number(_result['best'], 2, L.lang)} per token."}, _result["epochs"])),
            mo.md(f"""Your ModernGPT of {C.grouped(_result['params'], L.lang)} parameters trained for {len(_h)} {'epochs' if len(_h) > 1 else 'epoch'} on {_on}, in {C.number(_result['minutes'], 1, L.lang)} minutes, tokenizer and encoding included. Its validation loss went down to **{C.number(_result['best'], 2, L.lang)} per token** after epoch {_result['best_epoch']}: {C.number(_result['best_per_char'], 3, L.lang)} per character.{_versus}{_saved} {_verdict}{_who}{_story}"""),
            _course,
        ])
    trained_run = _result
    _out
    return (trained_run,)


@app.cell(hide_code=True)
def text_exercise_generate():
    mo.vstack([
        mo.md(r"""
        ### Exercise 4. Write with the cache

        Step 5 of the recipe. Complete `generate(model, ids, n)`: it writes `n` ids after `ids`, each time the likeliest next one, as greedy decoding does, with a cache for every block of `model`. The check gives it a small ModernGPT, compares its ids with those of the slow way, which sends all ids through the model at every step, and counts how many tokens the model gets per call.
        """),
        L.task("Make `generate` send the start once, and after that one new id per step, with the caches."),
    ])
    return


@app.cell(hide_code=True)
def exercise_generate():
    box_generate = L.code_box("generate", lines=18)
    box_generate
    return (box_generate,)


@app.cell(hide_code=True)
def check_generate(box_generate, result_gpt):
    result_gpt  # check again when exercise 3 saves your module
    result_generate = L.check("generate", box_generate.value)
    result_generate
    return (result_generate,)


@app.cell(hide_code=True)
def hints_generate():
    L.hints(
        "generate",
        [
            "Step 5 of the recipe: first a list with a new empty dict for every block in `model.blocks`, then the whole start through the model once, as a tensor of one row, with the caches.",
            "In the loop: the id with the largest logit at the last place, added to `out`; then only that id through the model, as a tensor of one row and one column, with the same caches.",
        ],
        why="The start goes through the model once and fills every cache; after that, a step costs one token per layer, and its query reads the keys and values of everything before it. The ids are those of the slow way, up to rounding.",
    )
    return


@app.cell(hide_code=True)
def demo_speed(result_generate, trained_run):
    result_generate, trained_run  # run again when exercise 4 saves your module, or after a run
    _s = C.speed()
    if _s is None:
        _out = mo.md("*Finish exercise 4 first: then your `generate` writes with your trained ModernGPT, or with the course's until you have trained one.*")
    elif isinstance(_s, str):
        _out = mo.md("*Writing stops with an error. Do your modules in `work/projects/llm/` still pass their checks? Delete a file you changed, and check its exercise again.*")
    else:
        _whose = "your ModernGPT of the demo above" if _s["source"] == "yours" else "the course's ModernGPT, trained on stories of Cosmopedia, until you have trained one"
        _same = "the same ids" if _s["same"] else "nearly the same ids: at some step the logits of two ids were so close that rounding picked the other one"
        _out = mo.md(f"""
        ### What your project can do now: write with a cache

        Your `generate` wrote {_s['tokens']} ids after "{C.B.START}" with {_whose}, in {C.number(_s['cache'], 2, L.lang)} seconds. The slow way, all ids through the model at every step, took {C.number(_s['full'], 2, L.lang)} seconds, and wrote {_same}. With a window of {_s['block']} tokens, the caches of its {_s['layers']} blocks hold {C.grouped(_s['numbers'], L.lang)} numbers; with as many key and value heads as query heads, {_s['h']} instead of {_s['kv_heads']}, they would hold {C.grouped(_s['numbers_mha'], L.lang)}.
        """)
    _out
    return


@app.cell(hide_code=True)
def text_exercise_gqa():
    mo.md(r"""
    ### Exercise 5. Why grouped-query attention?
    """)
    return


@app.cell(hide_code=True)
def choice_gqa_cell():
    _rope, _gqa = C.mean_best("rope"), C.mean_best("gqa")
    choice_gqa = L.choice(
        "gqa-why",
        question=f"In the course's runs, grouped-query attention, 2 key and value heads instead of 4, changes the validation loss from {C.number(_rope, 3, L.lang)} to {C.number(_gqa, 3, L.lang)} per token, on average. Why do Llama, Mistral and Qwen use it anyway?",
        options={
            "loss": "It lowers the loss of large models, even when it does not in small ones",
            "right": "It halves the KV cache while writing, and the loss barely changes",
            "flops": "It halves the FLOPs of attention while training",
        },
        correct="right",
        why={
            "loss": "Its paper does not claim a lower loss: it shows a quality close to that of multi-head attention, with a much smaller cache. What it saves is memory while writing: see [Grouped-query attention](/course/open/llm-gqa?lang=en).",
            "flops": "Every query head still compares its queries with all keys, so the scores take as many FLOPs as before; only the key and value projections get smaller. See [Grouped-query attention](/course/open/llm-gqa?lang=en).",
            "right": "With `h / kv_heads` query heads per key and value head, the cache is `h / kv_heads` times smaller: twice in the CPU plan, three times in the GPU plan, 4 or 8 times in Llama and Mistral, where the cache of long texts takes gigabytes.",
        },
    )
    choice_gqa
    return (choice_gqa,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Your GPT and its run of [Project: your GPT on your own tokens](/course/open/gpt-train-bpe?lang=en), with the parts of [RMSNorm](/course/open/llm-rmsnorm?lang=en), [Modern MLP blocks: GELU, SiLU and SwiGLU](/course/open/llm-swiglu?lang=en), [Rotary position embeddings (RoPE)](/course/open/llm-rope?lang=en) and [Grouped-query attention](/course/open/llm-gqa?lang=en), and the cache of [The KV cache](/course/open/llm-kv-cache?lang=en). Your project `llm` now has `modern_parts.py`, `modern_attention.py`, `modern_gpt.py` and `modern_generate.py`, and after the run a ModernGPT in `gpt_modern.pt`, with its merge rules in `gpt_modern.json`. The pretraining lessons ahead train this GPT on data you prepare yourself.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_parts, result_attention, result_gpt, result_generate, choice_gqa):
    L.wrap_up(result_parts, result_attention, result_gpt, result_generate, choice_gqa)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
