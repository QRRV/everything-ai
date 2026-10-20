# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-same-on-open-model. Route: course/route/lessen/taal.yaml. Checks and the course's run: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import qwen3
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-same-on-open-model", "en")
    C = L.checks
    R = C.course()          # the course's run on Qwen3-0.6B-Base, at the settings of a GPU
    Q = qwen3.header()["config"]


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Project: LoRA, fine-tuning with a low-rank update](/course/open/llm-lora?lang=en) and [Project: DPO, preferences without a reward model](/course/open/llm-dpo?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_frozen():
    quick_frozen = L.choice(
        "quick-frozen",
        question="After `add_lora(model, 8, 16)`, one step of training on a batch: what changes in the model?",
        options={
            "all": "Every weight, but each by a smaller step",
            "right": "Only the A and B of every LoRALinear",
            "last": "Only the output layer",
        },
        correct="right",
        why={
            "right": "Every weight the model had is frozen: `add_lora` turns `requires_grad` off for all of them, and only the new A and B get a gradient and a step.",
            "all": "`add_lora` freezes every weight the model had; the optimizer gets only the parameters that still need a gradient. See [Project: LoRA, fine-tuning with a low-rank update](/course/open/llm-lora?lang=en).",
            "last": "The adapters sit on the linear layers of every block, not on the output layer, and the weights of all those layers stay frozen. See [Project: LoRA, fine-tuning with a low-rank update](/course/open/llm-lora?lang=en).",
        },
    )
    quick_frozen
    return (quick_frozen,)


@app.cell(hide_code=True)
def quick_check_reference():
    quick_reference = L.choice(
        "quick-reference",
        question="Your `dpo_step` compares the policy with a reference model. In the DPO project, what was that reference model?",
        options={
            "copy": "A copy of the SFT model, loaded a second time next to the policy",
            "reward": "A reward model that gives every answer a score",
            "right": "The policy itself, with its adapters switched off",
        },
        correct="right",
        why={
            "right": "With every adapter off, the model is the model before DPO: one model in memory serves as both.",
            "copy": "That would work too, at the cost of a second model in memory; your `reference_logprob` switched the adapters off instead. See [Project: DPO, preferences without a reward model](/course/open/llm-dpo?lang=en).",
            "reward": "DPO needs no reward model: the reference model is the policy before training. See [Project: DPO, preferences without a reward model](/course/open/llm-dpo?lang=en).",
        },
    )
    quick_reference
    return (quick_reference,)


@app.cell(hide_code=True)
def text_why():
    mo.md(rf"""
    ## Why this matters

    Your model has {C.number(C.CAPSTONE['params'] / 1e6, 1, 'en')} million parameters and read stories for young children. Large teams publish far bigger models as open weights, with a license that says what you may do with them. Your LoRA, SFT and DPO do not belong to your model: they work on any model whose code you have. This lesson runs them on Qwen3-0.6B-Base, which read about 36 trillion tokens.

    **An everyday example.** A recipe from a cook abroad: the steps are the ones you know, but the units are grams instead of cups, and the herbs have other names. You translate those once, and then you cook as always.
    """)
    return


@app.cell(hide_code=True)
def predict_rope():
    guess_rope = L.guess(
        "rope",
        question=f"Qwen3 turns its queries and keys with RoPE, as your ModernGPT does, but it pairs dimension i with i + 64, where your `rope` pairs 2i with 2i + 1. With its own layout, its loss on {R['measured']['loss']} held-out answers of OASST1 is {C.number(R['rope']['half'], 2, 'en')}. The course loaded the same weights into code with your layout. What was the loss then?",
        options={
            "same": "About the same: the angles are the same, only the pairs differ",
            "bit": "A little higher, about 2",
            "far": "Far higher, above 4",
        },
        reveal=f"**Far higher: {C.number(R['rope']['pairs'], 2, 'en')}.** The weights learned which dimensions turn together; with other pairs, every query and key turns in a way the model never saw. No error, only wrong numbers. *An open model has the same parts as yours, but every detail must be its own.*",
    )
    guess_rope
    return (guess_rope,)


@app.cell(hide_code=True)
def text_idea():
    _ours = C.CAPSTONE
    mo.vstack([
        mo.md(rf"""
        ## The idea

        **What an open model is.** A few files on a model hub: `config.json` with its sizes, `model.safetensors` with every weight under a name, the tokenizer's files, and a license. Qwen3-0.6B-Base is Apache-2.0: you may use it, change it and share what you make, with its license, where it came from and what you changed.

        **The same parts.** Qwen3's blocks are those of [Project: upgrade your GPT to a modern LLM](/course/open/llm-modern-gpt?lang=en): RMSNorm, attention with RoPE and grouped-query attention, and SwiGLU. What differs is the sizes, and three details.

        | | your model | Qwen3-0.6B-Base |
        |---|---|---|
        | parameters | {C.number(_ours['params'], 0, 'en')} | {C.number(R['params'], 0, 'en')} |
        | blocks | {_ours['layers']} | {Q['num_hidden_layers']} |
        | width $d$ | {_ours['d']} | {Q['hidden_size']} |
        | heads, of which key-value | {_ours['heads']}, {_ours['kv_heads']} | {Q['num_attention_heads']}, {Q['num_key_value_heads']} |
        | numbers per head | {_ours['d'] // _ours['heads']}, $d$ divided by the heads | {Q['head_dim']}, its own number |
        | tokens in the vocabulary | {C.number(_ours['vocab'], 0, 'en')} | {C.number(Q['vocab_size'], 0, 'en')} |
        | tokens it trained on | {C.number(_ours['tokens'] / 1e6, 1, 'en')} million | about 36 trillion |
        """),
        mo.Html(C.sizes_picture({"title": ["parameters", "tokens it trained on"], "rows": ["your model", "Qwen3-0.6B"],
                                 "alt": f"Two bar charts on a log scale. Parameters: your model {C.number(_ours['params'] / 1e6, 1, 'en')} million, Qwen3-0.6B {C.number(R['params'] / 1e6, 0, 'en')} million. Tokens it trained on: your model {C.number(_ours['tokens'] / 1e6, 0, 'en')} million, Qwen3-0.6B about 36 trillion."})),
        mo.md(rf"""
        On these log scales: about 190 times the parameters, and more than half a million times the tokens. The three details:

        1. **QK-norm.** Before RoPE, the queries and keys of every head get an RMSNorm of their own, which keeps attention's scores steady while it trains.
        2. **RoPE's layout.** Dimension $i$ of a head turns with dimension $i + 64$, not with the one next to it.
        3. **Weight tying.** The output layer is the token embedding again, as in [Weight tying](/course/open/lm-weight-tying?lang=en).

        **The names.** The file names a weight as Qwen's code does, `model.layers.0.self_attn.q_proj.weight`; the course's Qwen3 uses your ModernGPT's names, `blocks.0.attention.query.weight`, so that your `add_lora` finds its layers. Loading is renaming every tensor, and checking every shape.

        **Tokens it never learned.** Qwen writes a conversation in ChatML, as in [Chat templates](/course/open/llm-chat-template?lang=en): `<|im_start|>user`, the question, `<|im_end|>`, then `<|im_start|>assistant` and the answer. But in the base model's embedding, `<|im_end|>` has exactly the numbers of `<|im_start|>` and {R['untrained']['tokens'] - 2} other rows. With weight tying they all get the same logit, so greedy writing picks the first of them, a text token. The one special token it learned is `<|endoftext|>`: the course ends every answer with that.

        ### New words and symbols

        | Word or symbol | What it means |
        |---|---|
        | **`config.json`** | the file of an open model that gives its sizes: width, blocks, heads, vocabulary |
        | **QK-norm** | an RMSNorm on the queries and on the keys of every head, before RoPE |
        | **untrained token** | a token the model never learned: its row of the embedding has exactly the numbers of other untrained rows |
        """),
        L.watch_out("Before you fine-tune a base model, check which special tokens it learned. An untrained token can never be more likely than its look-alikes, and adapters in the blocks do not change that: train its row of the embedding too, or end with a token the model knows."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Adapting an open model:

    1. Download its files at a fixed version; read its `config.json` and its license.
    2. Rename every tensor of its file to a name of your code, check every shape, and load.
    3. Write every question in its chat template, and end every answer with a token it learned.
    4. SFT with adapters: `add_lora`, `sft_example`, `sft_batch` and `pretrain_step`.
    5. Merge them with `merge_lora`; DPO with new adapters and `dpo_step`, the adapters off as the reference model.
    6. Measure on held-out questions and pairs, before and after every step.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The names and shapes of block 0, in Qwen3's file and in the course's model. The file's header is in the course, so this runs without the download. Run the cell.
    """)
    return


@app.cell
def example_names():
    _file = C.header()["tensors"]
    _ours = qwen3.model_names()
    print("in Qwen3's file:")
    for _name in _file:
        if _name.split(".")[:3] == ["model", "layers", "0"]:
            print("   ", _name, _file[_name]["shape"])
    print("in the course's model:")
    for _name in _ours:
        if _name.split(".")[:2] == ["blocks", "0"]:
            print("   ", _name, _ours[_name])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(rf"""
    The same 11 tensors per block, in another order and under other names: `self_attn.q_proj` is `attention.query`, `input_layernorm` is `norm1`, `q_norm` keeps its name. A weight is (outputs, inputs): the query of `[2048, 1024]` gives {Q['num_attention_heads']} heads of {Q['head_dim']}. The file has no output layer: it is the embedding again.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rename():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Rename the weights

        Complete `PARTS`, the parts of a name that differ, and `rename(name)`: it leaves out the first part, `model`, and gives every part in `PARTS` its new name; other parts, such as a block's number or `q_norm`, stay as they are. The check tries every kind of name in the file.
        """),
        L.task("Complete `PARTS` and `rename`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rename():
    box_rename = L.code_box("rename", lines=22)
    box_rename
    return (box_rename,)


@app.cell(hide_code=True)
def check_rename(box_rename):
    result_rename = L.check("rename", box_rename.value)
    result_rename
    return (result_rename,)


@app.cell(hide_code=True)
def hints_rename():
    L.hints(
        "rename",
        [
            "The six linear layers not in `PARTS` yet are `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` and `down_proj`; the worked example's second list has their names in the course's model.",
            "`name.split(\".\")` gives the parts; `[1:]` leaves out the first; `\".\".join(parts)` makes one name again.",
        ],
        why="The file and the code hold the same tensors under the names their authors chose; a table of the parts that differ turns the one into the other.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_params():
    mo.vstack([
        mo.md(rf"""
        ### Exercise 2. What LoRA trains on Qwen3

        Complete `lora_parameters(config, r)`: how many numbers your `add_lora` trains at rank `r` on a model with this `config.json`. It puts a LoRALinear on the seven linear layers of every block, and each trains `A`, `r` by its inputs, and `B`, its outputs by `r`. Read every size from `config`. The check tries Qwen3-0.6B, -1.7B and -4B, and a small model.
        """),
        L.task("Complete `lora_parameters`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_params():
    box_params = L.code_box("lora-parameters", lines=16)
    box_params
    return (box_params,)


@app.cell(hide_code=True)
def check_params(box_params):
    result_params = L.check("lora-parameters", box_params.value)
    result_params
    return (result_params,)


@app.cell(hide_code=True)
def hints_params():
    L.hints(
        "lora-parameters",
        [
            "The queries are `num_attention_heads` heads of `head_dim` numbers; the keys and the values are `num_key_value_heads` of them.",
            "For every layer, `r * n_in + n_out * r`; seven layers per block, and `num_hidden_layers` blocks.",
        ],
        why="LoRA trains two thin matrices per layer, so what it costs grows with the inputs plus the outputs of a layer, not with their product, as full fine-tuning does.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_end():
    mo.md(r"""
    ### Exercise 3. Where an answer ends
    """)
    return


@app.cell(hide_code=True)
def choice_end_cell():
    choice_end = L.choice(
        "end-id",
        question="Your `sft_example` puts `end_id` after every answer, and the chat loop stops at that token. Your `add_lora` leaves the embedding, which is also the output layer, frozen. Which token can SFT teach Qwen3-0.6B-Base to end its answers with?",
        options={
            "im_end": "`<|im_end|>`: SFT teaches it like any other token, as the template asks",
            "right": "`<|endoftext|>`, the special token that ended every text it read in pretraining",
            "any": "Either of the two, as long as every example ends with the same one",
        },
        correct="right",
        why={
            "right": f"The one special token the base model learned. `<|im_end|>` shares its row with {R['untrained']['tokens'] - 1} other tokens, so its logit is always theirs.",
            "im_end": f"`<|im_end|>` has exactly the numbers of {R['untrained']['tokens'] - 1} other tokens in the frozen output layer, so it can never be more likely than they are. Look at *Watch out* in *The idea* again.",
            "any": f"Not for this model: `<|im_end|>` shares its row of the frozen output layer with {R['untrained']['tokens'] - 1} other tokens. Look at *Watch out* in *The idea* again.",
        },
    )
    choice_end
    return (choice_end,)


@app.cell(hide_code=True)
def text_demo():
    _m, _k = R["measured"], C.RANK
    mo.vstack([
        mo.md(rf"""
        ### Train Qwen3 with your own steps

        The demo loads Qwen3 with your `rename`, puts adapters of rank {_k} on it with your `add_lora`, and trains them with SFT on OASST1: questions with the answer people ranked best. It merges them with your `merge_lora`, and trains new adapters with your `dpo_step` on HelpSteer2: two answers, of which people said which was better. It measures on held-out questions and pairs before and after.

        On a GPU it takes the course's steps: {C.SFT['steps']['gpu']} of SFT on {C.SFT['batch']['gpu']} questions and {C.DPO['steps']['gpu']} of DPO on {C.DPO['batch']['gpu']} pairs. On a CPU it takes {C.SFT['steps']['cpu']} and {C.DPO['steps']['cpu']}, on {C.SFT['batch']['cpu']} at a time: about {C.number(R['cpu']['minutes'], 0, 'en')} minutes and {C.number(R['cpu']['memory_gb'], 0, 'en')} GB, and so few steps of DPO stay at chance ({R['cpu']['dpo']['agree']} of {R['cpu']['dpo']['pairs']} held-out pairs). The run does not keep the model. While it trains, a chart shows the loss of every step of SFT, and then a second one under it shows that of DPO, with ln 2 dashed: the loss DPO starts at. The charts stay after the run. With so few questions in a step, the loss of a step jumps up and down; what SFT taught, the table under the charts measures on held-out questions.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B-Base (1.2 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*Qwen3-0.6B-Base is not on your computer yet. The button downloads it once from Hugging Face, from `Qwen/Qwen3-0.6B-Base` at a fixed version: 1.2 GB, its weights and its tokenizer. The exercises do not need it.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 1.2 GB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    mo.md("*Qwen3-0.6B-Base is downloaded.*")
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train Qwen3 with SFT and DPO")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_train(box_rename, result_rename, train_button):
    trained = None
    if result_rename.state != "correct":
        _out = mo.md("*Make `rename` right in exercise 1 first: the demo loads Qwen3 with it.*")
    elif not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif not train_button.value:
        _out = mo.md("*Press the button to train Qwen3.*")
    else:
        _phases = {"base": "Qwen3-0.6B-Base answers", "sft": "After SFT, it answers", "dpo": "After DPO, it answers", "agree": "Pairs measured"}

        _lives = {}   # the live chart of SFT, and of DPO once it starts

        def _show(_d):
            _now = _lives.get("dpo") or _lives.get("sft")
            if _d["phase"] == "load":
                mo.output.replace(mo.md("**Loading Qwen3** and measuring it …"))
            elif "step" in _d:
                if _d["phase"] not in _lives:
                    _lives[_d["phase"]] = L.live({"title": f"{_d['phase'].upper()}: the loss of every step", "x": "step", "y": "loss",
                                                  "alt": f"The loss of every step of {_d['phase'].upper()} so far" + ("; dashed, ln 2, the loss at the start." if _d["phase"] == "dpo" else ".")},
                                                 _d["steps"], low=0.0)
                    if _d["phase"] == "dpo":
                        _lives["dpo"].reference(C.DP.ln2_line(_d["steps"]), "ln 2")
                        _lives["dpo"].above = [mo.Html(_lives["sft"].picture())] if "sft" in _lives else []
                _lives[_d["phase"]].add(_d["step"], _d["loss"], f"**{_d['phase'].upper()}, step {_d['step']} of {_d['steps']}:** loss {C.number(_d['loss'], 3, 'en')}, {_d['seconds']:.0f} s")
            elif _now is not None:
                _now.line(f"**{_phases[_d['phase']]}** {_d['done']} of {_d['total']} …")
            else:
                mo.output.replace(mo.md(f"**{_phases[_d['phase']]}** {_d['done']} of {_d['total']} …"))

        _r = C.run(run_code(box_rename.value).names["rename"], _show)
        _charts = [mo.Html(_lives[_p].picture()) for _p in ("sft", "dpo") if _p in _lives]   # they stay after the run
        _out = mo.vstack(_charts + ([mo.md({
            "names": "*Your `rename` does not give every tensor of the file a name of the model, with the same shape. Does it still pass its check?*",
            "run": "*Loading Qwen3, or one of your modules in `work/projects/llm/`, stops with an error. Do your modules still pass their checks? Delete a file you changed, and check its exercise again.*",
            "lora": "*Your `add_lora` stops with an error on Qwen3, or leaves nothing to train. Does it still pass its check in the lesson about LoRA?*",
            "sft": "*Your `sft_example`, `sft_batch` or `pretrain_step` stops with an error on Qwen3, or the loss is no longer a number. Do they still pass their checks?*",
            "merge": "*Your `merge_lora` stops with an error on Qwen3. Does it still pass its check in the lesson about LoRA?*",
            "dpo": "*Your `dpo_step`, `answer_logprob` or `reference_logprob` stops with an error on Qwen3, or gives no number. Do they still pass their checks?*",
            "memory": "*The memory of your device is full. Close other programs, and press the button again.*",
        }[_r])] if isinstance(_r, str) else []))
        trained = None if isinstance(_r, str) else _r
    _out
    return (trained,)


@app.cell(hide_code=True)
def demo_result(trained):
    _run = trained or R
    _mine = trained is not None
    _b, _s, _d = _run["base"], _run["sft"], _run["dpo"]
    _whose = "Your run" if _mine else "The course's run"
    _stand = "" if _mine else "\n\n*The course's run stands in until you train Qwen3 yourself.*"

    def _answer(_text, _ended):
        _t = C.excerpt(_text, 400)
        return (_t or "*(nothing)*") + ("" if _ended else " *(no end yet)*")

    mo.vstack([
        mo.md(f"**{_whose}**, {_run['steps']['sft']} steps of SFT and {_run['steps']['dpo']} of DPO on {_run['device']}, {C.number(_run['seconds']['all'] / 60, 0, 'en')} minutes. LoRA trained {C.number(_run['trainable'], 0, 'en')} numbers.\n\n"
              f"| held out from training | Qwen3-0.6B-Base | after SFT | after DPO |\n|---|---|---|---|\n"
              f"| loss on the people's answers | {C.number(_b['loss'], 3, 'en')} | {C.number(_s['loss'], 3, 'en')} | |\n"
              f"| answers that stop by themselves | {_b['stopped']} of {_b['total']} | {_s['stopped']} of {_s['total']} | {_d['stopped']} of {_d['total']} |\n"
              f"| pairs on which DPO's reward agrees with the people | | | {_d['agree']} of {_d['pairs']} |{_stand}"),
        mo.md(f"**The first question it never trained on:** *{C.excerpt(_run['questions'][0], 300)}*\n\n"
              f"- Qwen3-0.6B-Base: {_answer(_b['texts'][0], _b['ended'][0])}\n"
              f"- after SFT: {_answer(_s['texts'][0], _s['ended'][0])}\n"
              f"- after DPO: {_answer(_d['texts'][0], _d['ended'][0])}"),
    ])
    return


@app.cell(hide_code=True)
def text_after_demo():
    _b, _s, _d = R["base"], R["sft"], R["dpo"]
    mo.md(rf"""
    **Reading the course's run.** SFT taught Qwen3 the form of a conversation: {_s['stopped']} of {_s['total']} answers now end by themselves, where {_b['stopped']} did. The loss on the people's answers fell only a little, from {C.number(_b['loss'], 2, 'en')} to {C.number(_s['loss'], 2, 'en')}: it already knew how such answers go.

    DPO moved it towards what people prefer: its reward agrees with the people on {_d['agree']} of {_d['pairs']} held-out pairs, where chance gives half, and on {_d['seen']['agree']} of {_d['seen']['pairs']} pairs it trained on. But only {_d['stopped']} of {_d['total']} answers now end within {C.NEW} tokens: the chosen answers are a little longer on average, and DPO is known to make answers longer. *Measure more than you train for.*
    """)
    return


@app.cell(hide_code=True)
def text_exercise_results():
    mo.md(r"""
    ### Exercise 4. What each step taught
    """)
    return


@app.cell(hide_code=True)
def explain_results():
    explain_results_cell = L.explain(
        "results",
        question="In the course's run, what did SFT teach Qwen3, and what did DPO teach it? And why did so few steps do, where your own model needed a thousand?",
        model_answer=f"SFT taught it the form of a conversation: to answer after the generation prompt and to stop with `<|endoftext|>` ({R['sft']['stopped']} of {R['sft']['total']} answers, where {R['base']['stopped']} did). It did not have to learn what to say: from about 36 trillion tokens it already knew the language, so {R['steps']['sft']} steps were enough and the loss fell only a little; my model had to learn the language and the task at once. DPO moved it towards what people prefer: {R['dpo']['agree']} of {R['dpo']['pairs']} held-out pairs against half by chance, and {R['dpo']['seen']['agree']} of {R['dpo']['seen']['pairs']} pairs it trained on, so it learned the pairs better than the preference. It also changed what it was not trained for: fewer answers stopped, {R['dpo']['stopped']} instead of {R['sft']['stopped']}. I would train on more pairs, and measure the length too.",
        points=[
            "SFT taught the form of a conversation, the answer and where it ends, not new knowledge",
            "a model pretrained on trillions of tokens needs few steps, because it already knows the language",
            "DPO's reward agrees with people on most held-out pairs, and on almost all it trained on",
            "DPO also changed what it was not trained for: fewer answers stopped, so measure more than you train for",
        ],
    )
    explain_results_cell
    return (explain_results_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The blocks of [Project: upgrade your GPT to a modern LLM](/course/open/llm-modern-gpt?lang=en), the adapters of [Project: LoRA, fine-tuning with a low-rank update](/course/open/llm-lora?lang=en) and the steps of [Project: DPO, preferences without a reward model](/course/open/llm-dpo?lang=en) worked on a model you did not train, once its names, details and template were its own. Before you share what you trained, name its sources: Qwen3-0.6B-Base and OASST1 under Apache-2.0, HelpSteer2 under CC BY 4.0.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_rename, result_params, choice_end, explain_results_cell):
    L.wrap_up(result_rename, result_params, choice_end, explain_results_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
