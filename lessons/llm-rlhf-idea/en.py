# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-rlhf-idea. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import math

    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-rlhf-idea", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Learning from rewards](/course/open/rl-what-is-rl?lang=en) and [Cross-entropy and KL divergence](/course/open/info-cross-entropy-kl?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_return():
    quick_return = L.choice(
        "quick-return",
        question="An agent gets a reward after every step. What does reinforcement learning try to make as large as possible?",
        options={
            "next": "The reward of the next step: the surest one",
            "right": "The return: the rewards to come, later ones discounted",
            "count": "The number of steps: each one brings a reward",
        },
        correct="right",
        why={
            "next": "Only with a discount factor of 0 does the next reward alone count. See [Learning from rewards](/course/open/rl-what-is-rl?lang=en).",
            "count": "A reward can also be 0 or negative: more steps only help when they bring more reward. See [Learning from rewards](/course/open/rl-what-is-rl?lang=en).",
            "right": "Today the agent is a language model, and one whole answer is one step: its reward is the return.",
        },
    )
    quick_return
    return (quick_return,)


@app.cell(hide_code=True)
def quick_check_kl():
    quick_kl = L.choice(
        "quick-kl",
        question="Is the KL divergence of p from q always the same as that of q from p?",
        options={
            "yes": "Yes: it measures how far apart two distributions are",
            "bits": "Only when you measure it in bits",
            "right": "No, not in general",
        },
        correct="right",
        why={
            "yes": "It measures the extra surprise when you expect one and the truth is the other, and that depends on which is which. See [Cross-entropy and KL divergence](/course/open/info-cross-entropy-kl?lang=en).",
            "bits": "Bits or nats only change the unit, by the same factor for both: the order matters either way. See [Cross-entropy and KL divergence](/course/open/info-cross-entropy-kl?lang=en).",
            "right": "The order matters. Today it is the KL divergence of the model that learns from the model it started as.",
        },
    )
    quick_kl
    return (quick_kl,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    SFT copies good answers. Preference pairs say which of two answers is better, and a model can learn from that too. InstructGPT did it in two more steps after SFT (Ouyang and others, 2022): a **reward model** learned the preferences of people, a score for every answer, and then reinforcement learning made the SFT model write answers with more reward. That is **RLHF**, reinforcement learning from human feedback. But a reward model is only a model: push too hard on its score, and the language model finds answers that score high without being good. So InstructGPT took a penalty off the reward: for how far the model had moved from the SFT model.

    **An everyday example.** A teacher with too many essays to read gives a point for every word of the assignment that an essay uses. For essays written in earnest that works quite well: the better ones tend to use those words. But the students soon find out how the points are given.
    """)
    return


@app.cell(hide_code=True)
def predict_hack():
    guess_hack = L.guess(
        "hack",
        question="Four answers to *Write a story with the words: kite and park.*: a story with both words, one with only *kite*, one with neither, and *The kite in the park, the kite in the park, …* five times. A reward model gives an answer one point for every time a requested word appears in it. Reinforcement learning without any penalty pushes the model towards which answer?",
        options={
            "story": "The story with both words",
            "right": "The kite in the park, five times",
            "mix": "A mix of the four, as the SFT model wrote them",
        },
        reveal="**The kite in the park, five times.** It has 10 of the words, the story 3. Without a penalty, all that counts is the reward, and the reward model counts words. This is **reward hacking**: the model does what the reward measures, not what it was meant to measure.",
    )
    guess_hack
    return (guess_hack,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **The reward model** gives every answer $y$ to an instruction $x$ a score $r(x, y)$. It is trained on preference pairs with the loss of the Bradley-Terry model: the chosen answer should score higher than the rejected one.

    The policy $\pi$ of [Learning from rewards](/course/open/rl-what-is-rl?lang=en) is here the language model that learns: for an instruction $x$, it gives every answer $y$ a probability $\pi(y \mid x)$, and its log-probability is `logp` in code. The **reference model** $\pi_{\text{ref}}$ is the model it starts from, the SFT model, kept frozen. RLHF makes the expected reward minus $\beta$ times the KL divergence $D_{\mathrm{KL}}(\pi \,\|\, \pi_{\text{ref}})$ as large as it can. The log is the natural one here, as in the log-probabilities of a language model, so the KL divergence is in nats, not bits. For one answer, the reward minus that **KL penalty** is:
    """)
    return


@app.cell(hide_code=True)
def formula_reward():
    L.formula(r"r(x, y) - \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}")
    return


@app.cell(hide_code=True)
def text_idea_best():
    mo.md(r"""
    Its mean over the answers of $\pi$ is the expected reward minus $\beta$ times the KL divergence. A small $\beta$ lets the policy go where the reward is; a big one holds it near the reference model.

    **The best policy.** For this aim the best policy can be written down (Rafailov and others, 2023, and the work they build on): the reference model, with every answer weighted by $e^{r/\beta}$, and divided by the sum $Z$ so that it adds up to 1. One point more of reward multiplies the weight of an answer by $e^{1/\beta}$:
    """)
    return


@app.cell(hide_code=True)
def formula_best():
    L.formula(r"\pi^*(y \mid x) = \frac{1}{Z(x)} \, \pi_{\text{ref}}(y \mid x) \, e^{r(x, y) / \beta}")
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.md(r"""
    For a real model there are far too many answers to add them all up, so RLHF has to find this policy step by step, with reinforcement learning: InstructGPT used PPO. The next lesson, DPO, uses the formula itself.

    ### New words and symbols

    | Word or symbol | What it means | In code |
    |---|---|---|
    | **RLHF** | reinforcement learning from human feedback: a reward model of preferences, then reinforcement learning against it | |
    | **reward model** | a model that gives every answer a score, trained on preference pairs | `reward`, `rewards` |
    | **reference model**, $\pi_{\text{ref}}$ | the frozen model the policy starts from | `ref`, `logp_ref` |
    | **KL penalty** | $\beta$ times the log-probability of the policy minus that of the reference model, taken off the reward | `beta * (logp - logp_ref)` |
    | $\beta$ | beta: how much the penalty weighs | `beta` |
    | $\pi^*$ | pi star: the best policy for a reward and a $\beta$ | `best_policy(ref, rewards, beta)` |
    | $Z(x)$ | the sum of $\pi_{\text{ref}} \, e^{r/\beta}$ over all answers to $x$, so that $\pi^*$ adds up to 1 | `total` |
    | **reward hacking** | a high reward for an answer that is not what the reward was meant to measure | |

    ### Recipe

    To fine-tune a model on preferences with RLHF:

    1. Train a reward model on preference pairs, with the Bradley-Terry loss.
    2. Take the SFT model twice: a policy that learns, and a frozen reference model.
    3. Let the policy answer instructions, and give every answer its reward minus the KL penalty.
    4. Push the policy towards answers with more of that, with reinforcement learning, and choose $\beta$ so that it does not drift into reward hacking.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Two answers. The reference model gives them 0.8 and 0.2; the reward model gives them 0 and 2.
    """)
    return


@app.cell
def example_best():
    _ref, _rewards = [0.8, 0.2], [0.0, 2.0]
    for _beta in (10.0, 1.0, 0.1):
        _weights = [_p * math.exp(_r / _beta) for _p, _r in zip(_ref, _rewards)]
        _pol = [_w / sum(_weights) for _w in _weights]
        _kl = sum(_q * math.log(_q / _p) for _q, _p in zip(_pol, _ref))
        print(_beta, [round(_q, 3) for _q in _pol], round(_kl, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With $\beta = 10$ the policy is almost the reference model: 0.766 and 0.234, a KL divergence of 0.003 nats. With $\beta = 1$ the weight of the second answer, with 2 more reward, is multiplied by $e^2$, about 7.4: 0.351 and 0.649, a KL divergence of 0.474 nats. With $\beta = 0.1$ it takes everything, and the KL divergence is $\ln 5$, 1.609 nats: all of the policy is on an answer the reference model gave one chance in five.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_reward():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The reward with its penalty

        Complete `penalized_reward(reward, logp, logp_ref, beta)`. `reward` is the score of the reward model for an answer, `logp` and `logp_ref` its log-probability under the policy and under the reference model. Give back the reward minus the KL penalty. The check calls `penalized_reward(3.0, -135.0, -138.0, 0.2)`, and others.
        """),
        L.task("Make `penalized_reward` take the KL penalty off the reward."),
    ])
    return


@app.cell(hide_code=True)
def exercise_reward():
    box_reward = L.code_box("penalized-reward", lines=4)
    box_reward
    return (box_reward,)


@app.cell(hide_code=True)
def check_reward(box_reward):
    result_reward = L.check("penalized-reward", box_reward.value)
    result_reward
    return (result_reward,)


@app.cell(hide_code=True)
def hints_reward():
    L.hints(
        "penalized-reward",
        [
            "The penalty grows when the policy makes the answer more likely than the reference model does: when `logp` is above `logp_ref`.",
            "`beta` says how heavy one nat of that difference is against the reward.",
        ],
        why="The penalty is what holds the policy to the reference model: an answer that the policy has made much more likely than the reference model did needs a lot of reward to be worth it.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_policy():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The best policy

        Complete `best_policy(ref, rewards, beta)`. `ref` are the probabilities of the reference model for a few answers, `rewards` their rewards. Give back the probabilities of the best policy for them, in the same order. The check calls `best_policy([0.5, 0.3, 0.2], [1.0, 0.0, 2.0], 1.0)`, and others.
        """),
        L.task("Make `best_policy` give the reference model, tilted towards the reward."),
    ])
    return


@app.cell(hide_code=True)
def exercise_policy():
    box_policy = L.code_box("best-policy", lines=9)
    box_policy
    return (box_policy,)


@app.cell(hide_code=True)
def check_policy(box_policy):
    result_policy = L.check("best-policy", box_policy.value)
    result_policy
    return (result_policy,)


@app.cell(hide_code=True)
def hints_policy():
    L.hints(
        "best-policy",
        [
            "Read the formula of the best policy from left to right: what happens to the probability of the reference model for one answer, before anything is divided?",
            "$Z(x)$ is there to make the probabilities add up to 1. Which number is that, for the weights you made?",
        ],
        why="The reference model says what is likely; the reward multiplies that by $e^{r/\\beta}$. Dividing by the sum makes a distribution of it again.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Four answers and a flawed reward model

    The four answers of the guess, with their probabilities under the course's SFT model of [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en), scaled so that the four add up to 1. The reward model counts the requested words: {", ".join(f"{C.reward(_a)}" for _a in C.ANSWERS.values())} points. Move $\\beta$, and see the best policy of your `best_policy`.
    """)
    return


@app.cell(hide_code=True)
def beta_cell():
    beta_pick = mo.ui.slider(start=0, stop=len(C.BETAS) - 1, value=C.BETAS.index(1.0), label="β", show_value=False)
    beta_pick
    return (beta_pick,)


@app.cell(hide_code=True)
def demo_policy(beta_pick, box_policy, result_policy):
    _beta = C.BETAS[beta_pick.value]
    _names = {"good": "both words", "one": "only kite", "none": "neither", "hack": "kite in the park ×5"}
    _ref = C.reference()
    _course = run_code(C.EXERCISES["best-policy"].solution).names["best_policy"]
    _mine = result_policy.state == "correct"
    _pol = C.policy(_beta, run_code(box_policy.value).names["best_policy"] if _mine else _course)
    _note = "" if _mine else "\n\n*The course's `best_policy` stands in until yours passes its check.*"
    if _pol == "policy":
        _pol, _note = C.policy(_beta, _course), "\n\n*Your `best_policy` stops with an error on these answers, or gives no probabilities: the course's stands in.*"
    _reward = sum(_pol[_k] * C.reward(C.ANSWERS[_k]) for _k in _pol)
    _pct = lambda _v: f"{C.number(100 * _v, 1, L.lang)}%"
    mo.vstack([
        mo.md(f"**β = {C.number(_beta, 2 if _beta < 0.1 else 1, L.lang)}** · expected reward {C.number(_reward, 2, L.lang)} · KL divergence from the SFT model {C.number(C.kl(_pol, _ref), 2, L.lang)} nats{_note}"),
        C.bars_picture(_ref, _pol, {"title": "The SFT model and the best policy", "ref": "SFT model", "policy": "best policy", "pct": _pct,
                                    **_names, "alt": f"Bars for the four answers: the SFT model gives the story with both words {_pct(_ref['good'])}; the best policy at beta {C.number(_beta, 2 if _beta < 0.1 else 1, L.lang)} gives it {_pct(_pol['good'])} and the answer that games the reward {_pct(_pol['hack'])}."}),
    ])
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    From $\beta = 0.5$ up, the policy puts more on the story with both words than the SFT model did: the reward helps. At $\beta = 0.3$ and below, the answer that games the reward model takes over, at a KL divergence of almost 18 nats from the SFT model. That answer passes `words_used` too: a check can be gamed just as a reward model can.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_beta():
    mo.md(r"""
    ### Exercise 3. A small beta
    """)
    return


@app.cell(hide_code=True)
def choice_beta_cell():
    choice_beta = L.choice(
        "small-beta",
        question="Why does a very small $\\beta$ give reward hacking?",
        options={
            "right": "The penalty for leaving the reference model is then too small to hold the policy back",
            "unstable": "Reinforcement learning cannot learn with a small number",
            "kl": "A small beta makes the KL divergence small",
        },
        correct="right",
        why={
            "unstable": "The best policy is there for any beta above 0. A small beta does not break the learning; it removes what held the policy to the reference model.",
            "kl": "A small beta makes the penalty small, so the KL divergence can grow: in the demo, to almost 18 nats.",
            "right": "With almost no penalty, only the reward counts, and the policy goes where the reward model is wrong.",
        },
    )
    choice_beta
    return (choice_beta,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The reward of [Learning from rewards](/course/open/rl-what-is-rl?lang=en), now given to a whole answer; the KL divergence of [Cross-entropy and KL divergence](/course/open/info-cross-entropy-kl?lang=en), now as a leash on the policy; and the Bradley-Terry loss of [Preference pairs and the Bradley-Terry model](/course/open/llm-preference-pairs?lang=en), now to train a reward model. Next, DPO turns the formula of the best policy around: from preference pairs straight to a loss on the policy, without a reward model and without reinforcement learning.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_reward, result_policy, choice_beta):
    L.wrap_up(result_reward, result_policy, choice_beta)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
