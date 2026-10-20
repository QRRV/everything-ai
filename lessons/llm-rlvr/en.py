# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-rlvr. Route: course/route/lessen/taal.yaml. Checks and the course's runs: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-rlvr", "en")
    C = L.checks
    R = C.COURSE_RUN          # the course's run: GRPO on its model of llm-dpo, 25 steps
    K = C.COURSE_KEEP         # keeping the answers that pass, and fine-tuning on them
    FIX = C.COURSE_REMEDIES   # two remedies for what the run took down


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Project: DPO, preferences without a reward model](/course/open/llm-dpo?lang=en) and [RLHF: learning from human preferences](/course/open/llm-rlhf-idea?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_when():
    quick_when = L.choice(
        "quick-when",
        question="In the DPO project, a checker turned answers of the SFT model into preference pairs. When did it look at the answers?",
        options={
            "right": "Once, before training, to make the pairs",
            "every": "After every step, on the newest answers of the model",
            "never": "Never: DPO needs no checker, only a reward model",
        },
        correct="right",
        why={
            "right": "The pairs were made once, from answers of the SFT model, and stayed the same during the whole run.",
            "every": "DPO trains on pairs made before the run; the answers the model gives while it learns are not checked. See [Project: DPO, preferences without a reward model](/course/open/llm-dpo?lang=en).",
            "never": "DPO needs no reward model, but it does need pairs, and in the project a checker made them. See [Project: DPO, preferences without a reward model](/course/open/llm-dpo?lang=en).",
        },
    )
    quick_when
    return (quick_when,)


@app.cell(hide_code=True)
def quick_check_kl():
    quick_kl = L.choice(
        "quick-kl",
        question="In RLHF, a KL penalty is taken off the reward of every answer. What is it for?",
        options={
            "accurate": "It makes the reward model more accurate",
            "short": "It makes the answers shorter, so that they cost less",
            "right": "It holds the policy near the reference model",
        },
        correct="right",
        why={
            "right": "The further the policy's log-probabilities move from those of the frozen reference model, the more is taken off: the policy stays near the model it started as.",
            "accurate": "The reward model is trained before, on preference pairs; the KL penalty does not change it. See [RLHF: learning from human preferences](/course/open/llm-rlhf-idea?lang=en).",
            "short": "It measures how far the policy moved from the reference model, not how long an answer is. See [RLHF: learning from human preferences](/course/open/llm-rlhf-idea?lang=en).",
        },
    )
    quick_kl
    return (quick_kl,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the DPO project a checker looked at the answers once, made pairs, and DPO learned from those pairs. But a program that can check an answer can do more: it can check every new answer while the model learns, and reward it at once. That is reinforcement learning with verifiable rewards, RLVR. It needs no reward model, which a policy can fool, and no people to compare answers: only a check a program can make, such as whether a word is left out, whether code passes its tests, or whether a sum has the right result. This is how reasoning models are trained: on maths and code problems whose answers a program can check.

    **An everyday example.** Practising free throws. You do not watch a video of good throws and copy them once; you throw, see at once whether the ball went in, and adjust the next throw. The basket is the program that checks.
    """)
    return


@app.cell(hide_code=True)
def predict_side():
    guess_side = L.guess(
        "side",
        question=f"The course trains its model of the DPO project with RL for {len(R['rewards'])} steps, on instructions to write a story without a word, rewarded by a program. On words the run never trained on, it leaves the word out {R['without']['after']} times out of {R['without']['total']} instead of {R['without']['before']}. Asked for a story *with* words, the kind of instruction DPO trained it on, how often does it now use every requested word?",
        options={
            "better": "More often than before",
            "same": "About as often as before",
            "worse": "Far less often",
        },
        reveal=f"**Far less often:** the model uses every requested word {R['with']['after']} times out of {R['with']['total']}, where it did {R['with']['before']} times before. The run raised what its reward checked, and nothing held up what it did not check. *What your program rewards, you get; what it does not measure, you can lose.*",
    )
    guess_side
    return (guess_side,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A verifiable reward.** A program looks at an answer and gives it a reward: here 1 when the story leaves the word out and has at least 3 sentences, else 0. The same program for every answer, during the whole run.

        **Groups, and how much better.** For every instruction the model writes a group of $G$ answers, here 8. Each gets its reward $r_i$, and its **advantage** is how much better it did than its group:
        """),
        L.formula(r"A_i = r_i - \frac{1}{G}\sum_{j=1}^{G} r_j"),
        mo.md(r"""
        An answer that passed in a group where most failed gets a large advantage; one that failed where most passed, a large negative one. A group whose answers all pass, or all fail, has advantages of 0: it teaches nothing, as an instruction whose answers all pass gave no DPO pairs.

        **The step.** The loss makes the answers with a positive advantage more likely, and those with a negative advantage less likely. Each answer counts with the mean log-probability $\overline{\log \pi}(a_i)$ of its tokens, so that a long answer counts as much as a short one:
        """),
        L.formula(r"\mathcal{L} = -\frac{1}{G}\sum_{i=1}^{G} A_i \,\overline{\log \pi}(a_i)"),
        mo.md(r"""
        This is GRPO, group relative policy optimization, in its simplest form: it needs no model that guesses how good an answer will be, because the group itself is the yardstick. Full GRPO also divides by the spread of the group's rewards, clips how far one step can change the probability of an answer, as PPO does, and adds a KL penalty to the reference model.

        **A simpler way, and why it was not enough here.** An older, simpler way to learn from a program is to keep only the answers that pass and fine-tune on them, as STaR and expert iteration do. The course tried that on its model: of {tried} stories, {kept} passed, and after SFT on them, it left the word out {after} times out of {total} instead of {before}. Keeping what passes never pushes down what fails, and here that was what it had to learn.

        ### New words and symbols

        | Word or symbol | What it means |
        |---|---|
        | **verifiable reward** | a reward a program gives by checking the answer, the same check for every answer |
        | **group** ($G$ answers) | the answers the model writes to one instruction, compared with each other |
        | **advantage** ($A_i$) | how much better an answer did than its group: its reward minus the group's mean |
        | **GRPO** | reinforcement learning that rewards every answer by its advantage within its group |
        | **expert iteration** | keeping only the answers that pass the program, fine-tuning on them, and doing that again |
        | $\overline{\log \pi}(a_i)$ | the mean log-probability of the tokens of answer $a_i$ under the policy |

        Each point is the mean reward of the 32 answers of one step: 4 instructions, 8 answers each.
        """.replace("{tried}", str(K["tried"])).replace("{kept}", str(K["kept"])).replace("{after}", str(K["without"]["after"]))
           .replace("{total}", str(K["without"]["total"])).replace("{before}", str(K["without"]["before"]))),
        mo.Html(C.rewards_picture(R["rewards"], {"title": "The course's run: mean reward per step", "x": "step", "y": "mean reward",
                                                 "course": "", "alt": "The mean reward of every step of the course's run of 25 steps: from about 0.3 in the first five steps to about 0.8 in the last five."})),
        L.watch_out("A program checks exactly what it checks, and the run finds every way to pass it. An empty story leaves every word out: that is why the course's reward also asks for at least 3 sentences. And whatever the program does not look at, the run may change: measure more than you reward."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    One step of GRPO with a verifiable reward:

    1. Take a few training instructions, and let the model write a group of answers to each, at its usual temperature.
    2. Give every answer its reward with the program.
    3. Give every answer its advantage: its reward minus the mean of its group.
    4. Take the loss: minus the mean of advantage times the mean log-probability of each answer; take a step.
    5. After the run, measure on instructions it never trained on: the kind you rewarded, and the other kinds too.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    One group of 8 stories without the word *park*: 3 passed the program, 5 did not. Run the cell.
    """)
    return


@app.cell
def example_group():
    _rewards = [1, 0, 0, 1, 1, 0, 0, 0]
    _mean = sum(_rewards) / len(_rewards)
    print("mean:", _mean)
    print("advantages:", [r - _mean for r in _rewards])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The mean is $3/8 = 0.375$. Each of the 3 stories that passed gets $1 - 0.375 = 0.625$, each of the 5 that failed $-0.375$; together they add up to 0. The step makes the 3 more likely and the 5 less likely, with as much weight in all: $3 \times 0.625 = 5 \times 0.375$.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_advantages():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The advantages of a group

        Complete `advantages(rewards)`: for every answer of a group, its reward minus the mean reward of the group. The check tries a group of 4, a group of 8 where only one passed, groups that all pass or all fail, and rewards between 0 and 1.
        """),
        L.task("Complete `advantages`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_advantages():
    box_advantages = L.code_box("advantages", lines=5)
    box_advantages
    return (box_advantages,)


@app.cell(hide_code=True)
def check_advantages(box_advantages):
    result_advantages = L.check("advantages", box_advantages.value)
    result_advantages
    return (result_advantages,)


@app.cell(hide_code=True)
def hints_advantages():
    L.hints(
        "advantages",
        [
            "The mean of a list is its sum divided by its length.",
            "A list comprehension gives every reward minus that mean.",
        ],
        why="An answer counts by how much better it did than the others to the same instruction, so the mean of the group is taken off every reward.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_loss():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The loss of a group

        Complete `policy_loss(logprobs, advantages)`. `logprobs` has, for every answer of a group, a tensor with the log-probabilities of its tokens under the model; `advantages` has a number for every answer. The loss is minus the mean, over the answers, of advantage times the mean log-probability of the answer's tokens. `logprobs[i].mean()` gives the mean of the tensor of answer `i`, and `torch.stack(terms)` makes one tensor of a list of numbers that are tensors, so that `.mean()` works on it too.

        The check tries an example of two answers of 2 and 4 tokens, the same two with equal advantages, a group of three answers, the gradient of the loss, and advantages of 0.
        """),
        L.task("Complete `policy_loss`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_loss():
    box_loss = L.code_box("policy-loss", lines=12)
    box_loss
    return (box_loss,)


@app.cell(hide_code=True)
def check_loss(box_loss):
    result_loss = L.check("policy-loss", box_loss.value)
    result_loss
    return (result_loss,)


@app.cell(hide_code=True)
def hints_loss():
    L.hints(
        "policy-loss",
        [
            "For answer `i` the term is its advantage times the mean of its log-probabilities. `advantages[i]` is its advantage, and the text of the exercise says how to take the mean of a tensor.",
            "The loss is minus the mean of the terms: a step that lowers it raises the terms with a positive advantage.",
        ],
        why="Lowering minus advantage times log-probability raises the log-probability of every answer with a positive advantage and lowers it for the others; the mean over the tokens keeps a long answer from counting more than a short one.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(rf"""
    ### Train your model with GRPO

    The button measures your model of the DPO project (or the course's, if you have none) on {C.MEASURE} instructions to leave out a word the run never trains on, and on {C.MEASURE} of the test instructions with words of the SFT project; then it trains it with GRPO, with your `advantages` and `policy_loss`, groups of {C.G} answers to {C.PROMPTS} instructions per step, for {C.STEPS['cpu']} steps on a CPU or {C.STEPS['gpu']} on a GPU; and then it measures again. That takes about seven minutes with the course's model on a CPU, less with a smaller model of your own, and much less on a GPU. Ten steps do not get as far as the course's {len(R['rewards'])}: its run is drawn under yours.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    demo_button = mo.ui.run_button(label="Train my model with GRPO")
    demo_button
    return (demo_button,)


@app.cell(hide_code=True)
def demo_train(box_advantages, box_loss, demo_button, result_advantages, result_loss):
    trained = None
    if result_advantages.state != "correct" or result_loss.state != "correct":
        _out = mo.md("*Make `advantages` and `policy_loss` right in exercises 1 and 2 first: the run trains with them.*")
    elif not demo_button.value:
        _out = mo.md("*Press the button to train your model.*")
    else:
        _live = L.live({"title": "Your run: mean reward per step", "x": "step", "y": "mean reward",
                        "alt": "The mean reward of every step of your run so far, as a line in ink with a dot at every step; dashed, the course's run on its model, at the same steps."}, C.steps(), role="ink", low=0.0, high=1.0)
        _live.reference(list(enumerate(R["rewards"][:C.steps()], 1)), "the course's run")

        def _show(_d):
            if _d["phase"] == "measure":
                _live.line(f"**Measuring** {_d['done']} of {_d['total']} …")
            elif _d["phase"] == "start":
                _live.line(f"**Training** · step 1 of {_d['steps']} …")
            else:
                _live.extend(_d["fresh"], f"**Step {_d['step']} of {_d['steps']}:** mean reward {C.number(_d['reward'], 2, 'en')}, {_d['seconds']:.0f} s")

        _r = C.run(run_code(box_advantages.value).names["advantages"], run_code(box_loss.value).names["policy_loss"], _show)
        trained = None if isinstance(_r, str) else _r
        if _r == "run":
            _out = _live.stopped("*Loading or running your model stops with an error. Do your modules in `work/projects/llm/` still pass their checks? Delete a file you changed, and check its exercise again.*")
        elif _r == "advantages":
            _out = _live.stopped("*Your `advantages` stops with an error on a group of the run, or does not give a number for every answer.*")
        elif _r == "loss":
            _out = _live.stopped("*Your `policy_loss` stops with an error on a group of the run, or does not give one number with a gradient.*")
        elif _r == "memory":
            _out = _live.stopped("*The device ran out of memory.*")
        else:
            _b, _a = _r["before"], _r["after"]
            _out = mo.vstack([
                mo.Html(C.rewards_picture(_r["rewards"], {"title": "Your run: mean reward per step", "x": "step", "y": "mean reward",
                                                          "course": "dashed: the course's run", "alt": "The mean reward of every step of your run, with the course's run of 25 steps as a dashed line."},
                                          course=R["rewards"])),
                mo.md(f"**{'Your model' if _r['source'] == 'yours' else 'The course’s model'}**, {_r['steps']} steps on {_r['device']}, {_r['seconds']:.0f} s:\n\n"
                      f"| instructions the run never trained on | before | after |\n|---|---|---|\n"
                      f"| without the word: left out | {_b['without']} of {_b['total']} | {_a['without']} of {_a['total']} |\n"
                      f"| with words: every word used | {_b['with']} of {_b['total']} | {_a['with']} of {_a['total']} |"),
            ])
    _out
    return (trained,)


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(rf"""
    **Reading your run.** In {C.STEPS['cpu']} steps *without* moves only a little, and on {C.MEASURE} instructions a change of one or two can be chance; *with words* can already go down. With the course's model on a CPU, your steps are the course's first {C.STEPS['cpu']}, so your line lies on the dashed one. The run does not save the model: your model of the DPO project stays as it was.
    """)
    return


@app.cell(hide_code=True)
def text_replay():
    mo.md(r"""
    ### Your run, step by step

    At every step, the run keeps the group of answers to the step's first instruction: what each answer got as its reward, and what your `advantages` made of it. Here you go through the steps: the mean reward of every step, with a line at the step you chose; beside it the advantage of every answer of that step's group as a bar up or down from zero; and under them the beginning of every answer.
    """)
    return


@app.cell(hide_code=True)
def replay_cell(trained):
    _run = trained if trained is not None else C.saved()
    if _run is None or not _run.get("groups"):
        _out = mo.md("*Once you have trained your model with GRPO, you can go through its run here.*")
    else:
        _labels = {"run": "Your run: mean reward per step", "run_x": "step", "run_y": "mean reward",
                   "title": "Step {step}: the group of one instruction", "x": "answer", "y": "advantage", "lang": L.lang,
                   "alt": "The mean reward of every step of your run, with a dashed line at the chosen step; and the advantage of every answer of the group of that step, as a bar up or down from zero.",
                   "step": "Step {step}", "mean": "mean reward of the group", "reward": "reward", "advantage": "advantage"}
        _out = L.explore("replay", "step", lambda _v: C.group_frame(_run, int(_v), _labels), start=C.replay_start(_run),
                         values=[_g["step"] for _g in _run["groups"]], role="",
                         task="Which answers of a group get a positive advantage, and what do the advantages of a group add up to?")
    _out
    return


@app.cell(hide_code=True)
def text_replay_after():
    mo.md(r"""
    The answers with a reward above the group's mean get a positive advantage, those below it a negative one, and in every group the advantages add up to zero: the step makes the ones above the mean more likely and the ones below less likely, the more so the further they are from the mean. A group in which every answer gets the same reward has an advantage of zero everywhere: the step learns nothing from it. GRPO learns only from groups in which some answers do better than others.
    """)
    return


@app.cell(hide_code=True)
def text_hack():
    mo.md(r"""
    ### Exercise 3. Why the guard
    """)
    return


@app.cell(hide_code=True)
def choice_hack_cell():
    choice_hack = L.choice(
        "hack",
        question="The course's reward is 1 only when the story leaves the word out *and* has at least 3 sentences. Why the second check?",
        options={
            "easier": "Stories of at least 3 sentences are easier for the program to check",
            "right": "Without it, an empty story would get the full reward",
            "model": "The model cannot write stories of fewer than 3 sentences",
        },
        correct="right",
        why={
            "right": "A story of no words, or of one short line, passes *leaves the word out* for any word. A run that finds such answers raises its reward without writing better stories: reward hacking, with a program instead of a reward model.",
            "easier": "The program checks a story of any length as easily. The point is what the run could do to pass the first check alone. Look at *Watch out* in *The idea* again.",
            "model": "It can: of the course's stories without a word, about one in six had only 2 sentences, and a run can push it to shorter ones. The point is what the run could do to pass the first check alone. Look at *Watch out* in *The idea* again.",
        },
    )
    choice_hack
    return (choice_hack,)


@app.cell(hide_code=True)
def text_exercise_trade():
    mo.md(rf"""
    ### Exercise 4. What the run took down

    In the course's run, leaving the word out went from {R['without']['before']} to {R['without']['after']} of {R['without']['total']}, and using every requested word from {R['with']['before']} to {R['with']['after']} of {R['with']['total']}. The course tried two remedies, {len(R['rewards'])} steps each:

    | run | without the word, of {R['without']['total']} | with words, of {R['with']['total']} |
    |---|---|---|
    | before | {R['without']['before']} | {R['with']['before']} |
    | GRPO on *without* | {R['without']['after']} | {R['with']['after']} |
    | GRPO on *without*, with a KL penalty, $\beta = 0.2$ | {FIX['kl']['without']} | {FIX['kl']['with']} |
    | GRPO with half the instructions *with* words, rewarded by their own check | {FIX['mix']['without']} | {FIX['mix']['with']} |
    """)
    return


@app.cell(hide_code=True)
def explain_trade():
    explain_trade_cell = L.explain(
        "trade",
        question="Why did *with words* go down, and what do the two remedies show? What would you try next?",
        model_answer=f"The reward only checked that the word was left out, so the run moved the model wherever that check pointed, and nothing held up using words when asked: it seems to have learned to leave requested words out more in general. A KL penalty of 0.2 held the policy so close to the reference model that it hardly learned what it was rewarded for: *without* stayed at {FIX['kl']['without']}; and *with* still went down, to {FIX['kl']['with']}, as the penalty only looks at the answers to the instructions the run trains on. Mixing in instructions with words, each with its own check, kept more of *with* ({FIX['mix']['with']}) while *without* rose less ({FIX['mix']['without']}). I would try a smaller beta, between 0 and 0.2, together with the mix, and measure both kinds, and other kinds of instructions, after every few steps.",
        points=[
            "the reward checked one kind of instruction, and nothing protected the other",
            "a KL penalty strong enough to hold the policy near the reference lets it learn little, and here it did not protect the other kind",
            "rewarding several kinds, each with its own check, spreads what the run learns",
            "measure every kind you care about, not only the one you reward",
        ],
    )
    explain_trade_cell
    return (explain_trade_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The checker of [Project: DPO, preferences without a reward model](/course/open/llm-dpo?lang=en) made pairs once; here a program checks every new answer and rewards it at once. The KL penalty and reward hacking of [RLHF: learning from human preferences](/course/open/llm-rlhf-idea?lang=en) came back, with a program in place of a reward model. The next lesson, *Did post-training help?*, measures your models on more kinds of instructions, which this run showed is needed.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_advantages, result_loss, choice_hack, explain_trade_cell):
    L.wrap_up(result_advantages, result_loss, choice_hack, explain_trade_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
