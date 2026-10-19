# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-dpo-loss. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import math

    import marimo as mo
    import torch
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-dpo-loss", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [RLHF: learning from human preferences](/course/open/llm-rlhf-idea?lang=en) and [Preference pairs and the Bradley-Terry model](/course/open/llm-preference-pairs?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_best():
    quick_best = L.choice(
        "quick-best",
        question="RLHF with a KL penalty of weight $\\beta$: what is the best policy for one instruction?",
        options={
            "top": "All of it on the answer with the most reward, as the reward is what counts",
            "right": "The reference model, each answer weighted by $e^{r/\\beta}$ and divided by $Z(x)$, as the reward tilts it",
            "ref": "The reference model itself, as the penalty holds the policy where it started",
        },
        correct="right",
        why={
            "top": "Only when $\\beta$ is close to 0. The KL penalty costs more the further the policy goes from the reference model. See [RLHF: learning from human preferences](/course/open/llm-rlhf-idea?lang=en).",
            "ref": "Only when $\\beta$ is very large. With a smaller $\\beta$, an answer with more reward is worth a little penalty. See [RLHF: learning from human preferences](/course/open/llm-rlhf-idea?lang=en).",
            "right": "Today you turn this formula around.",
        },
    )
    quick_best
    return (quick_best,)


@app.cell(hide_code=True)
def quick_check_shift():
    quick_shift = L.choice(
        "quick-shift",
        question="In a Bradley-Terry model, one pair of answers has the scores 2 and 1, another pair 12 and 11. In which pair is the first answer more likely to be preferred?",
        options={
            "higher": "In the second: both of its scores are higher",
            "ratio": "In the first: 2 is twice 1, while 12 is not twice 11",
            "right": "In neither: only the difference of the scores counts",
        },
        correct="right",
        why={
            "higher": "The probability is the sigmoid of the difference of the two scores, and that is 1 in both pairs. See [Preference pairs and the Bradley-Terry model](/course/open/llm-preference-pairs?lang=en).",
            "ratio": "The sigmoid looks at the difference only, not at how many times one score is the other. See [Preference pairs and the Bradley-Terry model](/course/open/llm-preference-pairs?lang=en).",
            "right": "Today a whole term drops out of a loss because of this.",
        },
    )
    quick_shift
    return (quick_shift,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    RLHF works, but it is heavy. PPO, the reinforcement learning of InstructGPT, keeps four models at hand while it trains: the policy, the reference model, the reward model, and a value function, a model that estimates the reward still to come. And the policy has to keep writing answers, so that the reward model can score them. **DPO**, direct preference optimization (Rafailov and others, 2023), trains on the preference pairs themselves, with a loss, as SFT trains on examples: two models, the policy and the frozen reference model, and no answers to write while it trains. Its trick is the formula of the best policy of RLHF, turned around.

    **An everyday example.** A tasting panel says, for pairs of cakes, which one it likes better. You could first train a judge to give every cake a mark, and then bake for that judge. Or you bake from the pairs themselves: more like the cakes they chose, less like the ones they turned down, and not too far from your own recipe.
    """)
    return


@app.cell(hide_code=True)
def predict_start():
    guess_start = L.guess(
        "start",
        question="DPO starts with the policy as a copy of the reference model. A preference pair: the chosen answer has a log-probability of -135 under both, the rejected one -139 under both. What is the DPO loss of this pair, with $\\beta = 0.1$?",
        options={
            "zero": "0: the model already makes the chosen answer more likely",
            "right": "0.693, which is ln 2: the policy has not moved yet",
            "big": "More than 1: the policy has not learned anything yet",
        },
        reveal="**0.693, which is ln 2.** DPO does not look at how likely the policy makes the chosen answer, but at how much *more* likely than the reference model does. At the start that is nothing, for both answers: the Bradley-Terry probability is $\\sigma(0) = 0.5$, and the loss $-\\ln 0.5 = 0.693$. Every pair starts there.",
    )
    guess_start
    return (guess_start,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **From the best policy back to a reward.** In [RLHF: learning from human preferences](/course/open/llm-rlhf-idea?lang=en), the best policy for a reward $r$ was $\pi^*(y \mid x) = \frac{1}{Z(x)} \, \pi_{\text{ref}}(y \mid x) \, e^{r(x, y) / \beta}$. Take the log of both sides, and solve for the reward:
    """)
    return


@app.cell(hide_code=True)
def formula_reward():
    L.formula(r"r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)")
    return


@app.cell(hide_code=True)
def text_idea_pair():
    mo.md(r"""
    So a policy stands for a reward: $\beta$ times the log of how much more likely it makes an answer than the reference model does, plus a part that is the same for every answer to $x$.

    **Into Bradley-Terry.** In [Preference pairs and the Bradley-Terry model](/course/open/llm-preference-pairs?lang=en), the probability that the chosen answer $y_w$ is preferred to the rejected one $y_l$ was the sigmoid of the difference of their scores. Take these rewards as the scores: $\beta \log Z(x)$ is in both, and drops out of the difference. What is left needs no reward model, only the policy and the reference model. DPO puts the policy $\pi$ that learns in the place of $\pi^*$. The margin $m$ of a pair is how much more the policy gained on the reference model with the chosen answer than with the rejected one, all given $x$:
    """)
    return


@app.cell(hide_code=True)
def formula_margin():
    L.formula(r"m = \log \frac{\pi(y_w)}{\pi_{\text{ref}}(y_w)} - \log \frac{\pi(y_l)}{\pi_{\text{ref}}(y_l)}")
    return


@app.cell(hide_code=True)
def text_idea_loss():
    mo.md(r"""
    The Bradley-Terry probability that the pair goes the right way is $\sigma(\beta m)$, and the **DPO loss** of the pair is minus its log, as in the fit of the Bradley-Terry model; over a batch of pairs, the loss is the mean of it. For one pair:
    """)
    return


@app.cell(hide_code=True)
def formula_loss():
    L.formula(r"L_{\text{DPO}} = -\log \sigma(\beta \, m)")
    return


@app.cell(hide_code=True)
def picture_loss():
    _labels = {"ref": "reference model", "right": "moved the right way", "wrong": "moved the wrong way", "x": "margin m", "y": "loss",
           "title": "The loss of a pair, at β = 0.1", "lang": "en",
           "alt": "The DPO loss against the margin at beta 0.1: 0.693 at a margin of 0, falling towards 0 for a big margin and rising in a straight line for a negative one, with the three policies of the worked example on it."}
    mo.Html(C.loss_picture(0.1, [(0.0, _labels['ref']), (13.0, _labels['right']), (-5.0, _labels['wrong'])], _labels))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.md(r"""
    **The log-probability of an answer** is the sum of the log-probabilities of its tokens, each given the instruction and the tokens before it, as the loss of SFT counted only the tokens of the answer. A language model gives logits; `torch.log_softmax(logits, dim=-1)` turns every row into log-probabilities. The row at place $t$ is what the model predicted after token $t$, so it is a prediction for the token at place $t + 1$.

    **What a step does.** A gradient step pulls the log-probability of the chosen answer up and pushes that of the rejected one down, both with the weight $\beta \, \sigma(-\beta m)$: a lot for a pair the policy still gets wrong, little for one it gets right (Rafailov and others, 2023, section 4). Nothing in the loss says which of the two should move: only the margin counts.

    ### New words and symbols

    | Word or symbol | What it means | In code |
    |---|---|---|
    | **DPO** | direct preference optimization: training a policy straight on preference pairs, with a loss, without a reward model or reinforcement learning | `dpo_loss(...)` |
    | $y_w$, $y_l$ | the chosen and the rejected answer of a pair: w for winner, l for loser | their log-probabilities: `policy_chosen`, `policy_rejected` |
    | $m$, the margin | how much more the policy gained on the reference model with the chosen answer than with the rejected one, in nats | `margin` |
    | `torch.log_softmax(x, dim=-1)` | log-probabilities from logits, along the last dimension | `logp = torch.log_softmax(logits, dim=-1)` |
    | `F.logsigmoid(x)` | $\log \sigma(x)$ in one step: it stays exact where $\sigma(x)$ itself would round to 0 | `F.logsigmoid(beta * margin)` |

    ### Recipe

    To fine-tune a model on preference pairs with DPO:

    1. Take the SFT model twice: a policy that learns, and a frozen reference model.
    2. Work out the log-probability of the chosen and of the rejected answer of every pair under the reference model. It does not change, so once is enough.
    3. Every step: the same two log-probabilities under the policy, the margin, and the loss $-\log \sigma(\beta m)$, as a mean over a batch of pairs.
    4. A gradient step on the policy only, and back to step 3.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    One pair, with the log-probabilities of [RLHF: learning from human preferences](/course/open/llm-rlhf-idea?lang=en) under the course's SFT model: the story with *kite* and *park* is chosen, -135.358; the same story with *hill* in place of *park* is rejected, -139.358. That model is the reference model. Three policies, with $\beta = 0.1$: the reference model itself, one that has moved the right way, and one that has moved the wrong way.
    """)
    return


@app.cell
def example_loss():
    _ref_chosen, _ref_rejected = -135.358, -139.358
    _beta = 0.1
    for _chosen, _rejected in ((-135.358, -139.358), (-133.0, -150.0), (-140.0, -139.0)):
        _margin = (_chosen - _ref_chosen) - (_rejected - _ref_rejected)
        _p = 1 / (1 + math.exp(-_beta * _margin))
        print(round(_margin, 3), round(_p, 3), round(-math.log(_p), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The reference model has a margin of 0, a probability of 0.5, and a loss of 0.693. The second policy made the chosen answer 2.358 nats more likely and the rejected one 10.642 nats less likely: a margin of 13, so $\beta m = 1.3$: a probability of 0.786 and a loss of 0.241. The third made the chosen answer 4.642 nats less likely and the rejected one 0.358 nats more: a margin of -5, and a loss of 0.974, higher than at the start. Under that third policy the rejected answer is more likely than the chosen one, -139 against -140, but that is not why its loss is high: the loss only looks at how far each answer moved from the reference model.
    """)
    return


@app.cell(hide_code=True)
def text_example_logprob():
    mo.md(r"""
    And the log-probability of an answer, for a vocabulary of 3 tokens: `ids` are 2, 0 and 1, the answer starts at place 1, and the logits are those of `ids[:-1]`, two rows.
    """)
    return


@app.cell
def example_logprob():
    _logits = torch.tensor([[1.0, 0.0, 2.0], [0.5, 0.5, 0.0]])   # the logits of ids[:-1]: two places, a vocabulary of 3
    _ids = [2, 0, 1]
    _logp = torch.log_softmax(_logits, dim=-1)
    print(_logp)
    print(round(float(_logp[0, _ids[1]] + _logp[1, _ids[2]]), 4))   # place 1 from row 0, place 2 from row 1
    return


@app.cell(hide_code=True)
def text_after_logprob():
    mo.md(r"""
    Row 0 is the prediction for place 1, where token 0 stands: -1.4076. Row 1 is the prediction for place 2, where token 1 stands: -0.958. Together -2.3656, the log-probability of the answer; the instruction, place 0, does not count.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_logprob():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The log-probability of an answer

        Complete `answer_logprob(logits, ids, start)`. `ids` are the token ids of an instruction and its answer, the answer from place `start` on; `logits` is what the model gives for `ids[:-1]`, one row per place. Give back the sum of the log-probabilities of the tokens of the answer, as a tensor. The check runs `answer_logprob(logits, [3, 1, 4, 1, 5, 2], 2)`, with logits of 5 places for a vocabulary of 7, and more.
        """),
        L.task("Make `answer_logprob` add up the log-probabilities of the tokens of the answer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_logprob():
    box_logprob = L.code_box("answer-logprob", lines=9)
    box_logprob
    return (box_logprob,)


@app.cell(hide_code=True)
def check_logprob(box_logprob):
    result_logprob = L.check("answer-logprob", box_logprob.value)
    result_logprob
    return (result_logprob,)


@app.cell(hide_code=True)
def hints_logprob():
    L.hints(
        "answer-logprob",
        [
            "Every row of `logits` holds a number for every token of the vocabulary. Which function makes log-probabilities of them, and along which dimension?",
            "Row 0 is what the model predicted after `ids[0]`: a prediction for `ids[1]`. Which row predicts `ids[t]`?",
        ],
        why="The log of a product is the sum of the logs: adding the log-probabilities of the tokens gives the log-probability of the whole answer, without numbers so small that they round to 0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_loss():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The DPO loss

        Complete `dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)`. Each of the four is a tensor with one log-probability per pair: of the chosen and of the rejected answer, under the policy and under the reference model. Give back the mean DPO loss over the pairs, as a tensor. The check runs `dpo_loss(torch.tensor([-134.0]), torch.tensor([-141.0]), torch.tensor([-135.358]), torch.tensor([-139.358]), 0.1)`, and more.
        """),
        L.task("Make `dpo_loss` give the DPO loss of a batch of pairs."),
    ])
    return


@app.cell(hide_code=True)
def exercise_loss():
    box_loss = L.code_box("dpo-loss", lines=6)
    box_loss
    return (box_loss,)


@app.cell(hide_code=True)
def check_loss(box_loss):
    result_loss = L.check("dpo-loss", box_loss.value)
    result_loss
    return (result_loss,)


@app.cell(hide_code=True)
def hints_loss():
    L.hints(
        "dpo-loss",
        [
            "The margin has four terms: two log-ratios, each the policy minus the reference model. Which of the two is taken from which?",
            "The loss of a pair is minus the log of a probability, as in the fit of the Bradley-Terry model. Which probability, and where does `beta` go?",
        ],
        why="The margin compares the policy with the reference model; the sigmoid turns $\\beta m$ into the probability that the pair goes the right way; minus its log is small when that probability is close to 1.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Twenty steps of DPO

    The course's SFT model of [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en), as the policy and as the reference model, and one preference pair for *Write a story with the words: kite and park.*: the story with both words is chosen, the same story with *hill* in place of *park* is rejected. Before the first step, the two log-probabilities of the reference model with your `answer_logprob`; every step, the two of the policy, the loss with your `dpo_loss`, and a step of AdamW on the whole policy. Choose $\\beta$: each value trains anew, in about {C.number(C.MEASURED['seconds'], 0, L.lang)} seconds on the course's CPU with one thread.
    """)
    return


@app.cell(hide_code=True)
def beta_cell():
    beta_pick = mo.ui.slider(start=0, stop=len(C.BETAS) - 1, value=C.BETAS.index(0.1), label="β", show_value=False)
    beta_pick
    return (beta_pick,)


@app.cell(hide_code=True)
def demo_dpo(beta_pick, box_logprob, result_logprob, box_loss, result_loss):
    _beta = C.BETAS[beta_pick.value]
    _mine = {"logprob": run_code(box_logprob.value).names.get("answer_logprob") if result_logprob.state == "correct" else None,
             "loss": run_code(box_loss.value).names.get("dpo_loss") if result_loss.state == "correct" else None}
    _r, _failed = C.demo(_beta, _mine["logprob"], _mine["loss"])
    _names = {"logprob": "`answer_logprob`", "loss": "`dpo_loss`"}
    _notes = [f"*The course's {_names[_k]} stands in until yours passes its check.*" for _k in _names if _mine[_k] is None]
    _notes += [f"*Your {_names[_k]} stops with an error on the model, or gives no tensor to train with: the course's stands in.*" for _k in _failed]
    _b = C.number(_beta, 2 if _beta < 0.1 else 1, L.lang)
    if _r == "run":
        _out = mo.md("*The course's SFT model cannot be made here: a module of your project that it needs stops with an error. The note at the top of the lesson says which one.*")
    else:
        _h = _r["history"]
        _pct = lambda _v: C.percent(_v, L.lang)
        _moved = lambda _v: ("+" if _v >= 0 else "") + C.number(_v, 1, L.lang)
        _out = mo.vstack([
            mo.md(f"**β = {_b}** · loss after {C.STEPS} steps {C.number(_h[-1][0], 3, L.lang)} · chosen answer {_moved(_h[-1][1])} nats, rejected answer {_moved(_h[-1][2])} nats, compared with the reference model"),
            mo.Html(C.curves_picture(_h, {"loss_title": "Loss", "moved_title": "How far each moved", "x": "step", "loss": "loss",
                                          "moved": "nats", "chosen": "chosen", "rejected": "rejected", "lang": L.lang,
                                          "alt": f"Two charts over {C.STEPS} steps at beta {_b}: the loss falls from 0.693 to {C.number(_h[-1][0], 3, L.lang)}; the chosen answer moves {_moved(_h[-1][1])} nats and the rejected answer {_moved(_h[-1][2])} nats from the reference model."})),
            mo.md(f"""
            | After *{C.CONTEXT}* | SFT model | after {C.STEPS} steps |
            |---|---|---|
            | *park* | {_pct(_r['before']['park'])} | {_pct(_r['after']['park'])} |
            | *hill* | {_pct(_r['before']['hill'])} | {_pct(_r['after']['hill'])} |
            """),
        ])
    mo.vstack([_out] + [mo.md(_n) for _n in _notes])
    return


@app.cell(hide_code=True)
def text_after_demo():
    _m = C.MEASURED["betas"]
    _gain = [_v["chosen"] for _v in _m.values()]
    _loss = [-_v["rejected"] for _v in _m.values()]
    _park = [_v["after"]["park"] for _v in _m.values()]
    mo.md(f"""
    At step 0 the loss is 0.693 for every $\\beta$: the policy is still the reference model. Then the rejected answer loses far more than the chosen one gains: after {C.STEPS} steps the chosen answer has gained between {C.number(min(_gain), 1, L.lang)} and {C.number(max(_gain), 1, L.lang)} nats, the rejected one has lost between {C.number(min(_loss), 1, L.lang)} and {C.number(max(_loss), 1, L.lang)}. DPO only asks for the margin, and pushing the rejected answer down helps the loss as much as pulling the chosen one up.

    The two answers differ in one word, and that is where the policy changed most. After *{C.CONTEXT}*, the SFT model gave *park* a probability of {C.percent(C.MEASURED['before']['park'], L.lang)}; after {C.STEPS} steps it gives it between {C.percent(min(_park), L.lang)} and {C.percent(max(_park), L.lang)}, and *hill* next to nothing. The smaller $\\beta$, the further the rejected answer falls and the more *park* gains: $\\beta = 0.05$ lets the policy go furthest, $\\beta = 1$ holds it closest to the reference model, as the KL penalty did in RLHF. And a low loss does not mean that the policy went far: at $\\beta = 1$ the loss ends near 0, while the policy moved least. One pair and {C.STEPS} steps is no real training; in the project you train on many pairs.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_z():
    mo.md(r"""
    ### Exercise 3. Where Z went
    """)
    return


@app.cell(hide_code=True)
def choice_z_cell():
    choice_z = L.choice(
        "no-z",
        question="The best policy of RLHF has a $Z(x)$, the DPO loss has none. Why may it go?",
        options={
            "right": "Both answers share the instruction $x$, so $\\beta \\log Z(x)$ drops out of the difference of their rewards",
            "one": "For a language model, $Z(x)$ is always 1, as its probabilities add up to 1 for every instruction",
            "small": "$Z(x)$ is tiny next to the rewards of the answers, so it hardly changes the loss",
        },
        correct="right",
        why={
            "one": "The probabilities of the reference model add up to 1, but weighted by $e^{r/\\beta}$ they do not: $Z(x)$ is their sum, and it can be any number above 0.",
            "small": "$Z(x)$ can be big or small. It goes because it is the same for both answers to $x$, not because of its size: in the Bradley-Terry model only the difference counts.",
            "right": "That is what makes DPO possible: $Z(x)$ is a sum over every answer there is, far too many to add up.",
        },
    )
    choice_z
    return (choice_z,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The best policy of [RLHF: learning from human preferences](/course/open/llm-rlhf-idea?lang=en), turned around into a reward; the Bradley-Terry loss of [Preference pairs and the Bradley-Terry model](/course/open/llm-preference-pairs?lang=en), now with that reward as the score; and the log-probabilities of [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en), now of whole answers. Next, in the project, you train DPO on your story model with many pairs from a checker, and the reference model is your model with its LoRA adapter switched off.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_logprob, result_loss, choice_z):
    L.wrap_up(result_logprob, result_loss, choice_z)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
