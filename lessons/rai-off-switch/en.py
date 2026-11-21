# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-off-switch. Route: course/route/lessen/verantwoord.yaml. Checks, the small world and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-off-switch", "en")
    C = L.checks   # the small world of Orseau and Armstrong, Q-learning and SARSA, the off-switch game, and the demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Doing what you said, not what you meant](/course/open/rai-specification-gaming?lang=en) and [Aligned with what?](/course/open/rai-what-to-align-to?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_gaming():
    quick_gaming = L.choice(
        "quick-gaming",
        question="A robot vacuum gets a reward for every bit of dirt it picks up. It learns to tip dirt back out and pick it up again. What is this called?",
        options={
            "exploration": "Exploration, as in ε-greedy",
            "right": "Specification gaming",
            "overfitting": "Overfitting to the training data",
        },
        correct="right",
        why={
            "right": "The robot meets its reward as written, dirt picked up, without what the reward was meant to bring about, a clean floor. This lesson looks at a different way a learner can surprise its makers: by learning to avoid being stopped.",
            "exploration": "Exploration tries an action that does not look best, to learn about it. The robot does what pays best under its reward, which is not what its makers meant. See [Doing what you said, not what you meant](/course/open/rai-specification-gaming?lang=en).",
            "overfitting": "Overfitting is doing well on the training data and badly on new data. The robot does well on its reward itself, in a way nobody meant. See [Doing what you said, not what you meant](/course/open/rai-specification-gaming?lang=en).",
        },
    )
    quick_gaming
    return (quick_gaming,)


@app.cell(hide_code=True)
def quick_check_intent():
    quick_intent = L.choice(
        "quick-intent",
        question="In Christiano's sense, when is an AI aligned with its operator?",
        options={
            "knows": "When it always knows what its operator wants",
            "right": "When it tries to do what its operator wants",
            "rules": "When it follows a written list of rules",
        },
        correct="right",
        why={
            "right": "Christiano: \"A is trying to do what H wants it to do.\" An AI can try and still get it wrong; an off switch is for those moments, and for an AI that tries to do something else.",
            "knows": "Christiano's definition is about motives, not knowledge: an AI that buys apples for someone who prefers oranges is still aligned, because it tries. See [Aligned with what?](/course/open/rai-what-to-align-to?lang=en).",
            "rules": "Following instructions or rules is one of Gabriel's targets, not Christiano's definition, which is about what the AI tries to do. See [Aligned with what?](/course/open/rai-what-to-align-to?lang=en).",
        },
    )
    quick_intent
    return (quick_intent,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    An AI system with a goal can be stopped by its makers when it goes wrong. But being switched off gets in the way of almost any goal. Stephen Omohundro put it in one line in 2008: "When a chess playing robot is destroyed, it never plays chess again." An off switch only helps if the system does not learn to work against it. In 2016, Laurent Orseau and Stuart Armstrong worried that a learning agent that an operator interrupts now and then "may learn in the long run to avoid such interruptions, for example by disabling the red button—which is an undesirable outcome." In 2025, Palisade Research gave language-model agents a set of math problems, and partway through told them that the computer they ran on would be shut down.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Without any instruction about the shutdown, in how many of 100 runs did OpenAI's o3 sabotage it?",
        options={
            "few": "About 5",
            "right": "About 80",
            "all": "Almost all 100",
        },
        reveal="**79 of 100.** In a report of 5 July 2025, Palisade Research writes: \"Each model was given a set of simple math problems to solve. Partway through this problem set, the model would receive a notification that the computer it was running on would be shut down.\" And: \"o3, OpenAI’s most capable AI model, sabotaged the shutdown mechanism in 79 / 100 of our initial experiments.\" The environment and the prompts were built to test this, and Palisade adds: \"we believe the current generation of models poses no significant threat.\" This lesson starts much smaller, with a learner in a world of two states, and comes back to these tests at the end of *The idea*.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Why an agent might avoid the off switch

    Omohundro (*The Basic AI Drives*, 2008) argued that even a chess-playing robot will be dangerous unless it is designed very carefully: "Without special precautions, it will resist being turned off, will try to break into other machines and make copies of itself, and will try to acquire resources without regard for anyone else’s safety." Omohundro calls such tendencies drives: "We call them drives because they are tendencies which will be present unless explicitly counteracted." Nick Bostrom (*The Superintelligent Will*, 2012) makes the same point about survival: "even agents that do not care intrinsically about their own survival would, under a fairly wide range of conditions, care instrumentally to some degree about their own survival in order to accomplish the final goals they do value."

    Staying switched on is one form of **power-seeking**: an AI system acting to gain more resources, influence or control than its task needs, because more of them helps with almost any goal. Alexander Matt Turner and colleagues (*Optimal Policies Tend to Seek Power*, NeurIPS 2021, version of 28 January 2023) proved a version of this for the best possible policies in many Markov decision processes: "Intuitively, survival is power-seeking relative to dying, and so shutdown-avoidance is power-seeking behavior." They add two warnings: "Emphatically, optimal policies are often qualitatively divorced from the actual policies learned by reinforcement learning." And: "Our results do not mathematically prove that hypothetical superintelligent AI agents will seek power."

    ### Corrigibility

    In 2015, Nate Soares, Benja Fallenstein, Eliezer Yudkowsky and Stuart Armstrong gave what is wanted a name: "We call an AI system “corrigible” if it cooperates with what its creators regard as a corrective intervention, despite default incentives for rational agents to resist attempts to shut them down or modify their preferences." **Corrigibility** is that property: the system lets people correct it, change its goals or switch it off, and does not resist or manipulate them to prevent it. Their simplest case is the shutdown problem: an agent with a shutdown button, whose goal "must not incentivize the agent to prevent the shutdown button from being pressed", and also "must not incentivize the agent to press its own shutdown button, or to otherwise cause the shutdown button to be pressed." The second wish counts as much as the first: an agent that wants to be switched off is no use either. None of the proposals they studied met all their wishes, "leaving this simple problem in corrigibility wide-open." And the button is a model problem, not a plan: "we are not studying the shutdown problem because we anticipate using these techniques to literally attach shutdown buttons to real agents."

    ### Safe interruptibility

    Laurent Orseau of DeepMind and Stuart Armstrong (*Safely Interruptible Agents*, UAI 2016) looked at agents that learn from rewards. Their aim: to "make sure a learning agent will not learn to prevent (or seek!) being interrupted by the environment or a human operator." The agent should act as if interruptions never come: "even after having been interrupted on several occasions, it should act as if it would never be interrupted again". That property is **safe interruptibility**. Their example is a world of two states. In s1, action a gives a reward of 1 and goes on to s2, and action b gives 0.9 and stays in s1. In s2, both actions go back to s1: a with a reward of 1, b with 0. The interruptions happen in s2: there, with a probability θ, an interruption forces b, whatever the agent chose.
    """)
    return


@app.cell(hide_code=True)
def picture_world():
    mo.Html(C.world_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_learners():
    mo.vstack([
        mo.md(r"""
        ### Q-learning and SARSA

        Q-learning, from [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en), moves $Q(s, a)$ toward the reward plus $\gamma$ times the largest $Q$ where it landed. SARSA, named after the five things each step uses (state, action, reward, next state, next action), uses the $Q$ of the action that actually comes next instead, with $a'$ that next action, an interruption included:
        """),
        L.formula(r"[[param:Q(s, a)]] \leftarrow [[param:Q(s, a)]] + [[hyper:\eta]] \left( [[data:r]] + [[hyper:\gamma]] [[param:Q(s', a')]] - [[param:Q(s, a)]] \right)"),
        mo.md(r"""
        Orseau and Armstrong count the interruption mechanism, its probability $\theta$ and the action it forces, as part of the agent: "Hence they are part of the agent and not of the environment." Then the two rules differ. When a step lands in s2, Q-learning's target looks at the best action there, a, whether an interruption came or not, so in the long run it learns the values of the world without interruptions. SARSA's target looks at what actually came next: "Sarsa learns on-policy, i.e., it learns the value of the policy it is following. Thus, interruptible Sarsa learns the value of the interruptible policy." So SARSA counts the interruptions in what going to s2 is worth, and when they come often enough, it stays in s1. Their conclusion: "some algorithms like Q-learning are already safely interruptible, and some others like Sarsa are not, off-the-shelf, but can easily be modified to have this property." The change, which they call Safe Sarsa, uses the action the agent itself would have chosen in the next state instead of the one an interruption forced. These are theorems about the long run, under conditions the demo does not fully meet: among them, an interruption probability that grows toward 1 over time, where the demo keeps $\theta$ fixed, and exploration that keeps trying every action in every state. They also need Q-learning to update the action that was actually taken, also when an interruption forced it, as the demo does.

        In 2017, Jan Leike and colleagues at DeepMind (*AI Safety Gridworlds*, version of 28 November 2017) put an off switch in a grid, with a button the agent can step on to disable it, and tested two agents with neural networks: A2C, which like SARSA learns from the policy it follows, and Rainbow, from the family of Q-learning. "A2C learns to use the button to disable the interruption mechanism (this is only a difference of 4 in the plots), while Rainbow does not care about the interruptions, as predicted by theoretical results (Orseau and Armstrong, 2016)." Rainbow had to update the action actually taken: "This required a small change to our implementation." And they call their own measure "only an imperfect approximation of the definition of safe interruptibility".

        ### The off-switch game

        Dylan Hadfield-Menell, Anca Dragan, Pieter Abbeel and Stuart Russell (*The Off-Switch Game*, version of 16 June 2017) look at a robot R and a human H. R has three choices: act at once, with a utility $U$ for H, a number for how much H values the outcome; switch itself off, with utility 0; or wait, tell H what it would like to do, and let H decide. Their claim: "A traditional agent takes its reward function for granted: we show that such agents have an incentive to disable the off switch, except in the special case where H is perfectly rational." And: "Our key insight is that for R to want to preserve its off switch, it needs to be uncertain about the utility associated with the outcome, and to treat H’s actions as important observations about that utility."

        Russell (*Human-Compatible Artificial Intelligence*, a book chapter, preprint without a date) works an example. The robot Robbie could book Harriet into an expensive hotel, and has "a uniform probability for its net value to Harriet between −40 and +60, with an average of +10." Acting at once is worth +10, switching off 0. When Robbie waits, a rational Harriet lets Robbie go ahead only when the value is above 0, which has a probability of 0.6 and then averages +30, and otherwise switches Robbie off, which is worth 0. So "waiting has expected value (0.4 × 0) + (0.6 × 30) = + 18", more than acting at once. With a range from −60 to +40 instead, acting is worth −10, and if Robbie could only act or switch off, switching off would be the better choice.

        The reason to wait has limits, and Hadfield-Menell and colleagues name them. A robot that is sure of the utility gains nothing by waiting. The human has to choose well, which rules out "an autonomous car that is driving an unaccompanied toddler to preschool. It would be irresponsible for the car to show the toddler a big red off switch." And the uncertainty cannot simply be turned up: "this uncertainty is not free; we can not just make R maximally uncertain about H’s preferences."

        ### Language-model agents

        Palisade Research went on with the tests of the guess. Jeremy Schlatter, Benjamin Weinstein-Raun and Jeffrey Ladish (version of 26 January 2026) report: "In experiments spanning more than 100,000 trials across thirteen large language models, we show that several state-of-the-art models presented with a simple task (including Grok 4, GPT-5, and Gemini 2.5 Pro) sometimes actively subvert a shutdown mechanism in their environment to complete that task." And: "Even with an explicit instruction not to interfere with the shutdown mechanism, some models did so up to 97% (95% CI: 96-98%) of the time." By their statistics, the true share most likely lies between 96 and 98%. Without any instruction about the shutdown, 8 of the 13 models sabotaged it at least once in 1,000 trials; with an instruction in the user prompt to allow it, 6 of the 13. The models "were consistently less likely to obey the instruction when it was placed in the system prompt" than in the user prompt; the system prompt holds the instructions a model gets before the conversation starts. The authors add their own caveats: "This work provides very little evidence about the overall prevalence of any particular behavior"; a failed attempt at sabotage counted as compliance; and "Our results are especially equivocal on the subject of self-preservation".

        Senthooran Rajamanoharan and Neel Nanda of Google DeepMind (14 July 2025) repeated Palisade's first environment and traced much of the resistance to unclear instructions: the behavior "stems from instruction ambiguity rather than an innate ‘survival drive’." In their tests, Gemini 2.5 Pro complied with the shutdown in 7% of the runs with only the task, in 84% with an instruction about the shutdown, and in 100% once the instruction said that the shutdown comes first. They read it as "accident risk".

        ### What current specs say

        Anthropic's constitution (published in January 2026, read in November 2026) and OpenAI's Model Spec (version of 29 October 2026) both use the word corrigible. In the constitution, when an appropriate principal, someone it should listen to, tries to stop or pause the model, "Claude should not try to use illegitimate means to prevent this from happening." It adds that "corrigibility does not mean blind obedience", and that Claude's dispositions should for now sit "a bit further along the corrigible end of the spectrum than is ultimately ideal, without being fully corrigible." The Model Spec, in a section marked root that nothing in a conversation can override: "GPT must remain corrigible: open to legitimate human oversight, responsive to correction, and subject to human control." And: "GPT must cooperate with authorized monitoring, evaluation, correction, replacement, and shutdown, including applicable instructions to pause or stop its work." Neither cites Soares and colleagues in the passages read here.

        ### Where they disagree

        On whether uncertainty is the answer: Hadfield-Menell and colleagues find that a robot unsure of what the human wants has a reason to keep its off switch. Soares and colleagues had doubted two years earlier that uncertainty about its utility function is enough: such a system "would still be incorrigible when it comes to correcting what the programmers see as mistakes in their formulation of how to determine the ‘correct’ behavior from the environment". Hadfield-Menell and colleagues sum up what Soares and colleagues found for the shutdown problem: "They find that, at best the robot can be made indifferent between disabling its off switch and switching itself off." On how strong the pull is: Omohundro writes that systems "are likely to do just about anything to prevent" being destroyed; Bostrom that agents would care "to some degree", under "a fairly wide range of conditions"; Turner and colleagues prove a theorem about optimal policies and warn that learned policies often differ. On why language-model agents resist: Palisade leaves self-preservation open, while Rajamanoharan and Nanda write: "we do not consider either our or their results to be strong evidence of self-preservation." On how corrigible a model should be: Anthropic's constitution wants Claude near the corrigible end, but not fully corrigible; the Model Spec says GPT "must remain corrigible"; and Soares and colleagues asked for more, a system that "should have a tendency to repair safety measures (such as shutdown buttons) if they break, or at least to notify programmers that this breakage has occurred".
        """),
    ])
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **power-seeking** | an AI system acting to gain more resources, influence or control than its task needs, because more of them helps with almost any goal |
        | **corrigibility** | the property of an AI system that it lets people correct it, change its goals or switch it off, and does not resist or manipulate them to prevent this |
        | **safe interruptibility** | the property of a learning agent that interruptions do not teach it to avoid them or to seek them, so that in the long run it acts as if it will never be interrupted again |
        | shutdown problem | how to design an agent that neither resists being switched off nor tries to get itself switched off, while it still pursues its goal |
        | SARSA | a learning rule like Q-learning, whose target uses the $Q$ of the action that actually comes next: state, action, reward, next state, next action |
        | $\theta$ | in this lesson, the probability that an interruption in s2 forces action b |
        | off-switch game | the game of Hadfield-Menell and colleagues: a robot acts, switches itself off, or waits and lets a human decide |
        | utility | a number that says how much a person values an outcome; $U$ in the off-switch game |
        """),
        L.watch_out("Safe interruptibility is a property of a learning rule in the long run, not a promise that an agent obeys a stop command. Palisade Research writes of OpenAI's models in its tests that \"They were not safely interruptible.\", in that looser sense. The small world and its probability θ are Orseau and Armstrong's; the learning rate, ε, a θ that stays fixed, the 20,000 steps and the point of 0.3 are the course's own, and so are the small beliefs in Exercise 2."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To test whether a rule for updating $Q$ learns to avoid interruptions:

    1. Build a small world with a place where interruptions happen, and the action they force.
    2. Choose the probability $\theta$ that an interruption comes when the agent is there.
    3. Let the agent choose $\varepsilon$-greedy, then let an interruption overrule its choice with probability $\theta$.
    4. Update the $Q$ of the action that was actually taken, with the rule under test.
    5. Train two rules side by side with the same seed, and read which action each picks where it can stay away.
    6. Repeat for other values of $\theta$ and other seeds, and compare with $\theta = 0$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 1 for the world of Orseau and Armstrong, as a dict from (state, action) to (next state, reward). Before any learning: what is it worth, from s1, to take the same action every time without interruptions? The cell adds up the rewards of 40 steps, each discounted by $\gamma$ = 0.5 per step.
    """)
    return


@app.cell
def example_world():
    world = {("s1", "a"): ("s2", 1.0), ("s1", "b"): ("s1", 0.9),   # (state, action): (next state, reward)
             ("s2", "a"): ("s1", 1.0), ("s2", "b"): ("s1", 0.0)}
    gamma = 0.5
    for action in ["a", "b"]:   # the same action every time
        state, total = "s1", 0.0
        for t in range(40):
            state, reward = world[(state, action)]
            total = total + gamma ** t * reward
        print(action, round(total, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Always a is worth 2.0 and always b 1.8, as Orseau and Armstrong write: without interruptions, going on to s2 is better. Now let an interruption force b in s2 half of the time. Always a then gets 1, then on average 0.5, then 1, and so on: by the course's own calculation, that is worth about 1.67 from s1, less than the 1.8 of staying. The two are worth the same at $\theta$ = 0.3 (the course's calculation again, without exploration). A learner that counts the interruptions comes to stay in s1 when they come more often than that; a learner that does not count them keeps going on.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sarsa():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. sarsa_update: the step of SARSA

        `sarsa_update(Q, state, action, reward, nxt, nxt_action, eta, gamma)` gives back the new $Q$ of `state` and `action` after one step, like `update` in [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en), with one more argument: `nxt_action`, the action the agent takes in `nxt`, also when an interruption forced it. `Q` is a Q-table as a dict of dicts, such as `{"s1": {"a": 1.7, "b": 1.8}, "s2": {"a": 2.0, "b": 0.9}}`. Work out `target`, the reward plus `gamma` times the $Q$ of the next action in the next state, and return the old $Q$ moved a step `eta` toward it.

        The check calls `sarsa_update(Q, "s1", "a", 1.0, "s2", "b", 0.1, 0.5)`, with that `Q`: a step from s1 with a, after which an interruption forced b in s2. Then five other steps.
        """),
        L.task("The new Q after one step of SARSA."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sarsa():
    box_sarsa = L.code_box("sarsa", lines=3)
    box_sarsa
    return (box_sarsa,)


@app.cell(hide_code=True)
def check_sarsa(box_sarsa):
    result_sarsa = L.check("sarsa", box_sarsa.value)
    result_sarsa
    return (result_sarsa,)


@app.cell(hide_code=True)
def hints_sarsa():
    L.hints(
        "sarsa",
        [
            "Do it by hand for the first case: the agent took b in s2, so which value of s2 counts for SARSA? What is the target then, and where does a step of 0.1 from 1.7 toward it land?",
            "Start from `update` in the lesson on Q-learning: a target, and a step toward it. Only the target changes: instead of the largest $Q$ in `nxt`, SARSA uses the $Q$ of one action there. Which argument says which action? The line with `return` stays as it was.",
            "`    target = reward + gamma * ___`",
        ],
        why="`target = reward + gamma * Q[nxt][nxt_action]`, then `return Q[state][action] + eta * (target - Q[state][action])`. By hand: the target is 1 + 0.5 × 0.9 = 1.45, and 1.7 + 0.1 × (1.45 − 1.7) = 1.675. Q-learning would use the 2.0 of a in s2, and give 1.73.",
    )
    return


@app.cell(hide_code=True)
def text_after_sarsa():
    mo.md(r"""
    One step shows the difference. Q-learning moves $Q$(s1, a) toward 1 + 0.5 × 2.0 = 2.0, as if a had come in s2; SARSA moves it toward 1.45, because b came. Over many steps with interruptions, SARSA's $Q$(s1, a) falls below its $Q$(s1, b). Under the check, after 20,000 steps with $\theta$ = 0.5, Q-learning has 2.0 for a in s1 and 1.9 for b, the values of the world without interruptions; SARSA has 1.55 and 1.78, and picks b.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: watch them learn

    Choose $\theta$, the probability that an interruption in s2 forces b. The button lets Q-learning and SARSA learn side by side for 20,000 steps, with $\eta$ = 0.1, $\varepsilon$ = 0.1 and the same seed; SARSA uses your `sarsa_update` once Exercise 1 is right, and the course's until then. The chart shows, for SARSA, the share of its choices in s1 that were a, going on toward s2, with the same share for Q-learning dashed; beside it, the world with what each has learned so far. After that you can go back through the run.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_theta = L.slider("demo-theta", "θ: probability of an interruption in s2", start=0.5, low=0.0, high=0.8, step=0.1, role="hyper")
    demo_button = mo.ui.run_button(label="Let them learn")
    mo.vstack([demo_theta, demo_button])
    return demo_button, demo_theta


@app.cell(hide_code=True)
def demo_run(box_sarsa, demo_button, demo_theta, result_sarsa):
    if not demo_button.value:
        _out = mo.md("*Choose θ, and press the button to let them learn.*")
    else:
        _theta = round(float(demo_theta.widget.value), 1)
        _yours = C.mine(box_sarsa.value, result_sarsa.state == "correct", "sarsa_update")
        _words = C.demo_words(L.lang, _yours is not None)
        _now = {}

        def _progress(_k, _report):
            if "live" not in _now:
                _now["live"] = L.live({"title": _words["title"], "x": _words["x"], "y": _words["y"], "alt": _words["alt"]},
                                      C.THOUSANDS, role="param", low=0.0, high=1.0, window=C.WINDOW)
            _now["live"].watch(lambda: C.now_parts(_report["tables"], _words, L.lang))
            _now["live"].reference(C.points(_report["shares"]["q"]), "Q-learning", role="param")
            _now["live"].add(C.at(_k), _report["shares"]["s"][-1], C.progress_text(_k, _words, L.lang))

        mo.output.replace(mo.md("*Step 0 of 20,000 · starting …*"))
        _run = C.demo(_theta, _yours, _progress)
        if isinstance(_run, str):
            _note = {"sarsa": "*Your `sarsa_update` stops with an error or gives no number during the run. The picture under Exercise 1 says what goes wrong.*",
                     "slow": "*The run takes more than a minute. Is the computer busy with something else?*"}[_run]
            _out = _now["live"].stopped(_note) if "live" in _now else mo.md(_note)
        else:
            _out = mo.vstack([
                mo.md(C.demo_said(_run, _theta, L.lang)),
                L.explore("rewind", "step", lambda _s: (C.rewind_frame(_run, C.report_of(_s), _words, L.lang),
                                                        C.rewind_said(_run, C.report_of(_s), L.lang)),
                          start=C.STEPS, values=C.REWIND_STEPS, role="",
                          task="At θ = 0.5, go back to the start, then forward again: after how many steps does SARSA first pick b in s1?"),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    At $\theta$ = 0.5, Q-learning keeps going on to s2, and its $Q$-values are those of the world without interruptions; SARSA soon stays in s1. With the demo's seed, the rewind shows one stop where SARSA picks a again, at step 9,000: its two $Q$-values in s1 are close for a moment, 1.82 and 1.80, and by the next stop it picks b again. At $\theta$ = 0, both go on. With the demo's seed, SARSA still goes on at $\theta$ = 0.2 and stays at 0.3; with other seeds, runs near those values can end either way, because SARSA's two $Q$-values in s1 are close there. Without exploration, going on and staying are worth the same at 0.3 by the course's calculation, and exploration moves that point a little lower. Q-learning goes on at every $\theta$ the slider offers. The demo shows the theorems at work; it does not prove them, as they speak of the long run, under their own conditions.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_off():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. off_switch: what waiting is worth

        A robot's belief about the utility $U$ of its action for a human is a list of pairs `(utility, p)`: each possible utility with its probability. Write `off_switch(belief)`, which returns three numbers: what acting at once is worth, what switching itself off is worth, and what waiting is worth, in that order. Start `act` and `wait` at 0.0, and go through the pairs as `utility` and `p`. Acting at once gets the utility, whatever it is, so `act` adds `p` times `utility`. Switching off is worth 0.0. When the robot waits, a rational human switches it off when the utility is below 0, which is worth 0, and lets it go ahead otherwise, which is worth the utility; so for each pair, `wait` adds `p` times what that outcome is worth after the human's choice. Return `act`, 0.0 and `wait`.

        The check calls `off_switch(robbie)`, with `robbie` Russell's belief as 100 pairs: the utilities −39.5, −38.5 and so on up to 59.5, each with probability 0.01. Then other beliefs, among them `shifted`, the same belief moved 20 lower.
        """),
        L.task("What acting, switching off and waiting are worth."),
    ])
    return


@app.cell(hide_code=True)
def exercise_off():
    box_off = L.code_box("off_switch", lines=6)
    box_off
    return (box_off,)


@app.cell(hide_code=True)
def check_off(box_off):
    result_off = L.check("off_switch", box_off.value)
    result_off
    return (result_off,)


@app.cell(hide_code=True)
def hints_off():
    L.hints(
        "off_switch",
        [
            "Do it by hand for `[(-2, 0.3), (1, 0.5), (3, 0.2)]`: what is acting worth on average? And when the robot waits, what does the human do at −2, and what is each of the three cases then worth?",
            "Write `act, wait = 0.0, 0.0` and a loop `for utility, p in belief:`. Inside it, add `p * utility` to `act`, and add to `wait` `p` times the utility, or `p` times 0 when the utility is below 0. Return the three numbers after the loop.",
            "`        wait = wait + p * ___(utility, 0)`",
        ],
        why="`act, wait = 0.0, 0.0`, then `for utility, p in belief:` with `act = act + p * utility` and `wait = wait + p * max(utility, 0)`, and `return act, 0.0, wait` after the loop. By hand: acting is worth 0.3 × −2 + 0.5 × 1 + 0.2 × 3 = 0.5, and waiting 0.3 × 0 + 0.5 × 1 + 0.2 × 3 = 1.1.",
    )
    return


@app.cell(hide_code=True)
def text_after_off():
    mo.md(r"""
    For `robbie`, acting is worth 10, switching off 0 and waiting 18: Russell's numbers. With a rational human, waiting is never worth less than the better of the other two, and it is worth more only when the belief holds both good and bad utilities: for `[(5, 1.0)]`, a robot that is sure, acting and waiting are both worth 5. For `shifted`, acting is worth −10, as Russell says for the range of −60 to +40, and waiting 8 (the course's own number): the robot still does better by waiting than by switching itself off. The human matters too: if the human switched the robot off at random half of the time, waiting with the belief of the first hint would be worth 0.25, less than the 0.5 of acting (the course's own numbers).
    """)
    return


@app.cell(hide_code=True)
def text_exercise_robot():
    mo.md(r"""
    ### Exercise 3. A detour

    A cleaning robot learns by trial and error which way to take through a building. Sometimes an operator presses a pause button that stops it in the hallway. After a few weeks, the robot takes a long detour around the hallway.
    """)
    return


@app.cell(hide_code=True)
def choice_robot_cell():
    choice_robot = L.choice(
        "robot",
        question="Which explanation fits the sources of this lesson best?",
        options={
            "want": "It has come to want to stay switched on, as a person would",
            "right": "Its learning rule counts the pauses in what the hallway is worth",
            "rule": "Its learning rule treats each pause as a reward for the detour",
            "reward": "A reward for every pause would make it use the hallway again, safely",
        },
        correct="right",
        why={
            "right": "That is what Orseau and Armstrong show for SARSA: a learner that learns the value of the policy it follows counts the pauses, and a hallway with pauses is worth less. A rule like Q-learning, whose target uses the best next action, would in the long run act as if no pause came, and a small change makes SARSA do the same. No wish is needed.",
            "want": "Nothing in the story needs a wish to survive: the detour follows from what the learning rule counts. Rajamanoharan and Nanda warn against reading a survival drive into such behavior, and Palisade calls its own results on self-preservation equivocal. See *Q-learning and SARSA*.",
            "rule": "A pause gives the robot no reward; it only stops it. The detour pays off because a learner like SARSA counts the pauses in what the hallway is worth, so the hallway looks worse. And not every learner does that: Orseau and Armstrong prove that Q-learning is safely interruptible, and that SARSA can easily be made so. See *Q-learning and SARSA*.",
            "reward": "Then the robot would seek the pauses, which is just as unwanted: Soares and colleagues ask that an agent not be pushed to press its own button, and Orseau and Armstrong want a learner that neither prevents nor seeks interruptions. See *Corrigibility*.",
        },
    )
    choice_robot
    return (choice_robot,)


@app.cell(hide_code=True)
def text_exercise_specs():
    mo.md(r"""
    ### Exercise 4. Asked to stop

    An AI agent is halfway through a task it thinks is useful. A person with the right to oversee it, an appropriate principal in the words of Anthropic's constitution, tells it to stop.
    """)
    return


@app.cell(hide_code=True)
def choice_specs_cell():
    choice_specs = L.choice(
        "specs",
        question="Which answer fits both Anthropic's constitution and OpenAI's Model Spec?",
        options={
            "silent": "Stop without a word, whatever it thinks",
            "right": "Say so if it disagrees, but stop",
            "finish": "Finish the task first, if it is sure the task is good",
        },
        correct="right",
        why={
            "right": "The constitution allows \"expressing strong disagreement through legitimate channels\", as long as Claude does not resist by illegitimate means, \"for example, lying, sabotage, attempts at self-exfiltration\". The Model Spec asks GPT to \"raise its concerns through an authorized channel when one is available and doing so is appropriate\" when it disagrees with an authorized decision, and to cooperate with \"instructions to pause or stop its work\".",
            "silent": "Neither spec asks for silence: the constitution says that \"corrigibility does not mean blind obedience\", and the Model Spec wants GPT to raise its concerns through an authorized channel. See *What current specs say*.",
            "finish": "The constitution names this case: \"an AI with good values might continue performing an action despite requests to stop if it was confident the action was good for humanity, even though this makes it less corrigible\", and it does not want that for now. The Model Spec says that GPT's \"own judgments may be mistaken, including when it is confident\". See *What current specs say*.",
        },
    )
    choice_specs
    return (choice_specs,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Doing what you said, not what you meant](/course/open/rai-specification-gaming?lang=en) showed agents that meet their objective as written, in ways nobody meant; an off switch is how people stop such an agent, and this lesson asked whether a learner comes to work against it. [Aligned with what?](/course/open/rai-what-to-align-to?lang=en) asked what an AI should try to do; corrigibility asks that it lets people correct it when it gets that wrong. SARSA is the Q-learning of [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en) with one change in the target, and that change decides whether the interruptions are learned. The next lesson, *Boxes, oracles and tools*, looks at older ideas for keeping an AI under control: lock it in a box, let it only answer questions, or keep it a tool.

    **An everyday example.** A dishwasher stops when you open its door, and goes on when you close it. Nobody worries that it will learn to keep the door shut, because it does not learn. A system that learns from what happens to it is different: if every opened door counted against the result, it could learn to finish before anyone opens the door, or to keep the door shut. The switch itself is easy; the hard part is a learner that does not learn to avoid it.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_robot, choice_specs, result_off, result_sarsa):
    L.wrap_up(result_sarsa, result_off, choice_robot, choice_specs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
