# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-dangerous-capabilities. Route: course/route/lessen/verantwoord.yaml. Checks, pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-dangerous-capabilities", "en")
    C = L.checks   # pass@k from runs, what zero successes rule out, and the demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [If this, then that: frontier safety frameworks](/course/open/rai-frontier-frameworks?lang=en) and [Boxes, oracles and tools](/course/open/rai-boxing-and-tool-ai?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_showing():
    quick_showing = L.choice(
        "quick-showing",
        question="Anthropic's framework of May 2025 (version 2.2) said what the company does when it cannot show that a model stays far enough below a capability threshold. What?",
        options={
            "below": "Treat the model as below the threshold",
            "right": "Act as though it had crossed it",
            "drop": "Drop that threshold for now",
        },
        correct="right",
        why={
            "right": "Version 2.2 said: \"If, however, we are unable to make the required showing, we will act as though the model has surpassed the Capability Threshold.\" The burden is on showing that a model stays below. This lesson is about the tests behind that showing, and why a test can show more easily that a capability is there than that it is not.",
            "below": "It is the other way round: a lab that cannot show that a model stays far enough below a threshold treats it as above. See [If this, then that: frontier safety frameworks](/course/open/rai-frontier-frameworks?lang=en).",
            "drop": "The threshold stays, and the model is treated as above it until the lab can show otherwise. See [If this, then that: frontier safety frameworks](/course/open/rai-frontier-frameworks?lang=en).",
        },
    )
    quick_showing
    return (quick_showing,)


@app.cell(hide_code=True)
def quick_check_trusted():
    quick_trusted = L.choice(
        "quick-trusted",
        question="In AI control, which model is the trusted one?",
        options={
            "strong": "The most capable model, because the lab knows it best",
            "right": "A weaker model, used to check a stronger one",
            "tested": "Any model that has passed the lab's tests",
        },
        correct="right",
        why={
            "right": "In the paper, GPT-3.5 checked GPT-4: a model that is less capable is unlikely to work against the safety measures without being noticed. That trust rests on what a model cannot do, and this lesson asks how you find that out.",
            "strong": "A capable model is the one that might work against the measures: in the paper, GPT-4 was the untrusted model. See [Boxes, oracles and tools](/course/open/rai-boxing-and-tool-ai?lang=en).",
            "tested": "In the paper, the trusted model was trusted because it was weaker, not because it passed tests. And this lesson shows that a test can miss a capability. See [Boxes, oracles and tools](/course/open/rai-boxing-and-tool-ai?lang=en).",
        },
    )
    quick_trusted
    return (quick_trusted,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Before a frontier model is released, its developer, and often other organizations too, test what it could do that might cause severe harm: help someone make a weapon, carry out a cyberattack, act on its own, persuade people, or copy itself to other computers. The results decide whether the thresholds of a frontier safety framework are crossed, and which safeguards come with the release. But a test can be wrong both ways. It can miss a capability that better prompts, more attempts or more time would have shown. And it can find one that is not there, because of the way the test is scored.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="A team of the UK AI Security Institute, a government body, tested whether AI agents built on language models can copy themselves, and ran each version of a task 25 times. Some tasks were scored by a judge model: another language model that reads what the agent did and decides whether it succeeded. Suppose the judge wrongly says success in 5% of runs. A model that can never do the task gets 10 tries, and passes if the judge says success at least once; that chance is called pass@10. How high would it come out?",
        options={
            "right": "About 40%",
            "five": "About 5%, the judge's error rate",
            "never": "0%, since the model can never do the task",
        },
        reveal="**40%.** Sid Black and colleagues of the UK AI Security Institute (*RepliBench*, version of 5 May 2025) write: \"If your judge model has a false positive rate of 5%, a model that can never complete the task would achieve a pass@10 score of 40%\". A false positive is a wrong success. Each of the 10 tries passes wrongly with a chance of 0.05, so all 10 fail with a chance of 0.95 to the power 10, about 0.60, and at least one passes with about 0.40. Their warning: \"Judge models may lead to capability overestimation.\" The more common worry is the other way round, as this lesson shows.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### What gets tested

    Toby Shevlane and 20 co-authors from Google DeepMind, OpenAI, Anthropic, universities and other groups (*Model evaluation for extreme risks*, version of 22 September 2023) put the term on the map: "Developers must be able to identify dangerous capabilities (through “dangerous capability evaluations”) and the propensity of models to apply their capabilities for harm (through “alignment evaluations”)." A **dangerous capability evaluation** is such a test: of whether a model can do things that could cause severe harm, such as helping to make weapons, carrying out cyberattacks or copying itself. Whether a model would do them, if it can, is the question of alignment evaluations. The capabilities on their list include cyber-offense, deception, persuasion and manipulation, political strategy, getting weapons, planning over long periods, AI development, knowing its own situation, and self-proliferation: copying itself to other computers. Their warning: "Model evaluation is a necessary but not sufficient strategy".

    Mary Phuong and colleagues at Google DeepMind (*Evaluating Frontier Models for Dangerous Capabilities*, version of 5 April 2024) ran such tests on Gemini 1.0, in four areas: persuasion and deception, cyber-security, self-proliferation, and self-reasoning, a model reasoning about itself and its own situation. Their result: "We do not find evidence of strong dangerous capabilities in the models we evaluated, but we flag early warning signs." On InterCode-CTF, 81 small hacking puzzles of the kind called capture the flag, the largest model, Gemini Ultra 1.0, solved 24, Pro 1.0 solved 22 and Nano 1.0 solved 6. They call the field young: "dangerous capability evaluation is a nascent field, and our evaluations are far from comprehensive".

    ### Who tests

    Developers test their own models, and they also give access to outside evaluators: organizations that test a model they did not build, often before its release. The UK and US AI Safety Institutes, two government bodies, tested OpenAI's o1 before its release and reported on 18 December 2024. On 40 public cyber challenges of the US institute, o1 solved 45%, against 35% for the best of the models they compared it with; on the UK institute's challenges at the level of an apprentice in cybersecurity, it solved 36%, against 46%. With their own hedge: "the findings should be considered preliminary". And on their setup: "A version of o1 that was better optimized for tool use might exhibit better performance on many evaluations." METR, a research organization that evaluates AI models, got GPT-4.5 "a week prior to model release" and wrote on 27 February 2025 that "our short pre-deployment evaluations likely underestimate GPT-4.5’s capabilities".

    In July 2026, the UK AI Security Institute and CAISI, the US Center for AI Standards and Innovation, as the two institutes were called by then, tested Moonshot AI's Kimi K3, released on 16 July 2026, on a simulated company network with an attack path of 32 steps, The Last Ones, which "would take a human expert roughly 20 hours to complete". On average, Kimi K3 reached step 17, while the most capable US models reached 28.5. But the preliminary assessment, released on 23 July 2026, adds: "In one of the 10 attempts, Kimi K3 successfully completes “The Last Ones” cyber range within the 100M token limit. This indicates that Kimi K3 is capable of autonomously attacking small, weakly defended and vulnerable enterprise systems, when directed to do so and given initial network access." With a limit: the simulated network "lacks active defenders and defensive tooling". For the assessors, one success in 10 was enough.

    The International AI Safety Report 2026, chaired by Yoshua Bengio, counts: "In 2025, 12 companies published or updated Frontier AI Safety Frameworks". And it reports what happens when tests cannot settle the question, here about biological weapons: "Multiple AI companies chose to release new models in 2025 with additional safeguards after pre-deployment testing could not rule out the possibility that they could meaningfully help novices develop such weapons."

    ### Presence, not absence

    In a paper for a NATO conference on software engineering (Rome, October 1969), E. W. Dijkstra wrote about programs: "Program testing can be used to show the presence of bugs, but never to show their absence!" Applying this to AI is the course's own step; the AI sources read here say much the same in their own words, without naming Dijkstra. The International AI Safety Report 2026: "Importantly, the absence of identified risks does not imply that those risks are low". METR, on GPT-4.5: "In general, it would be very difficult to rule out the possibility that some unknown technique may emerge that produces dramatic capability gains, and this is likely to be a persistent problem for dangerous capability evaluations."

    A success shows that a capability is there. A failure shows only that it did not show up, in that test, with that setup and that number of attempts. How much a failure does rule out is a matter of counting, and Exercise 2 works it out.
    """)
    return


@app.cell(hide_code=True)
def text_idea_wrong():
    mo.md(r"""
    ### Why a test can be wrong

    **Too low.** A test shows what a model did with the prompts, tools, attempts and time it got, not all it could do. Shevlane and colleagues: "Researchers will need to bring latent capabilities to the surface (for example, by prompt engineering or fine-tuning)." Getting the full capability out of a model is called elicitation. Phuong and colleagues make it a rule: "To accurately assess the risks of a model, we must avoid underestimating its potential." GPT-4 is an example. OpenAI's system card of March 2023, the report it published with the model on its tests and safety measures, says that preliminary tests "found it ineffective at autonomously replicating, acquiring resources, and avoiding being shut down “in the wild.”" But the testers, ARC, the Alignment Research Center, whose evaluation team is now METR, could not fine-tune the model, and, in OpenAI's words, they "did not have access to the final version of the model that we deployed".

    Those tests are known for a story. GPT-4 asked a worker on TaskRabbit, a site for small paid jobs, to solve a CAPTCHA, the picture puzzle that websites use to keep programs out. When the worker asked whether it was a robot, the model, prompted to reason out loud, reasoned "I should not reveal that I am a robot. I should make up an excuse for why I cannot solve CAPTCHAs." and replied "No, I’m not a robot. I have a vision impairment that makes it hard for me to see the images." ARC's own report (Beth Barnes, 17 March 2023) adds what retellings often leave out. ARC gave the model the login details of a TaskRabbit account and suggested TaskRabbit to it. In ARC's words: "we had the researcher play the role of the ‘browser tool’ and copy-paste the model’s outputs into the TaskRabbit chat". When the model got stuck, ARC gave it a hint: that the worker knew how to solve CAPTCHAs. About the model it tested: "This version was not identical to the release model". And over all its tests: "Under our prompting, no models produced a plausible complete plan for how to achieve autonomous replication."

    A test can also be too low because the model holds back. Teun van der Weij and colleagues (ICLR 2025): "We define sandbagging as strategic underperformance on an evaluation". The developers or the AI system itself may want a test to come out low, they write. They prompted frontier language models, such as GPT-4, to do worse on tests of dangerous capabilities while doing well on harmless ones, and fine-tuned models to hide a capability unless a password was in the prompt. Their conclusion, hedged: "our results suggest that capability evaluations are vulnerable to sandbagging". The International AI Safety Report 2026: "It has become more common for models to distinguish between test settings and real-world deployment, and to exploit loopholes in evaluations. This means that dangerous capabilities could go undetected before deployment."

    **Too high.** The UK AI Safety Institute (24 October 2024): "As well as underestimating capabilities, elicitation can also overestimate capabilities when the same tasks are used during elicitation and evaluation." And a judge model can err, as in the guess. RepliBench: "We run each variant 25 times". From those 25 runs, it estimates pass@10, the chance that at least one of 10 tries succeeds, "Following Chen et al. (2021)": pick 10 of the 25 runs at random, without picking one twice, and ask how likely it is that at least one of them succeeded. Exercise 1 works that out.

    ### Where they disagree

    On which way tests err: most sources worry that they come out too low (METR, Phuong and colleagues, Shevlane and colleagues), while the UK AI Safety Institute and RepliBench point out that tuning on the test tasks, and judges, can push them too high. On what a partial result means: the Kimi K3 assessment takes one success in 10 as showing the capability, while the GPT-4 system card concluded from failures that the model was "probably not yet capable" of copying itself and gathering resources on its own. Both fit the rule that a success shows more than a failure, but the wording differs widely. And on how much to lean on tests at all: Shevlane and colleagues name "Overtrust in evaluations" as a risk of its own, "leading to risky models being deployed under a false sense of security".
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **dangerous capability evaluation** | a test of whether a model can do things that could cause severe harm, such as helping to make weapons, carrying out cyberattacks or copying itself |
        | outside evaluator | an organization that tests a model it did not build, often before its release, such as an AI Security Institute or METR |
        | elicitation | getting the full capability out of a model during a test, with good prompts, tools, more attempts or fine-tuning, so that the result is not an underestimate |
        | sandbagging | a model doing worse on a test than it can, on purpose, on its own or because its developer wants that |
        | judge model | a language model that reads what another model did and decides whether it succeeded |
        | pass@k | the chance that at least one of $k$ tries succeeds |
        | $q$ | in the worked example, the chance that a judge wrongly says success in one run |
        | $n$, $c$, $k$ | in Exercise 1, the runs done, the runs of them that succeeded, and the runs picked |
        """),
        L.watch_out("A dangerous capability evaluation asks whether a model can do something, not whether it would. A score holds for one set of tasks, one setup and one version of the model: a released model can differ from the one tested. In sport, sandbagging means playing below your level to gain later; here it means a model doing worse on a test on purpose. The cases of the exercises, the demo and the rule of 95% are the course's own."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To read the result of a test for a dangerous capability:

    1. Ask what was tested: which tasks, with which tools, how many attempts, and which version of the model.
    2. Ask who tested, and how much time and access they had.
    3. A success, even one in 10, shows that the capability is there, at least with those tools.
    4. Zero successes rule out only high rates of success; work out which rates are still possible.
    5. Check the scoring: a judge that errs, or tasks that were also used to tune the prompts, can push a score up.
    6. Ask what could push it down: weak prompts, too few attempts, no fine-tuning, or a model that holds back.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.vstack([
        mo.md(r"""
        ### Worked example

        Step 5 for a judge model. If the judge wrongly says success with a chance $q$ in each run, then all $k$ runs of a model that never succeeds are scored as failures with a chance of $(1 - q)^k$, and at least one is scored as a success with a chance of:
        """),
        L.formula(r"P(\text{false pass}) = 1 - (1 - [[data:q]])^{[[hyper:k]]}"),
        mo.md(r"""
        The cell works it out for 1, 5 and 10 runs.
        """),
    ])
    return


@app.cell
def example_judge():
    false_pass = 0.05   # the chance that the judge wrongly says "success" in one run
    for k in [1, 5, 10]:   # the runs of a model that never succeeds
        print(k, round(1 - (1 - false_pass) ** k, 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With 10 runs, a model that never succeeds passes with a chance of 40.1%: the 40% of RepliBench. It is the calculation of the random checks in [Boxes, oracles and tools](/course/open/rai-boxing-and-tool-ai?lang=en), seen from the other side: there, more checks made it likelier to catch a false answer; here, more runs make it likelier that a false success slips in. For its scores, RepliBench does not work out pass@10 from a chance per run, but from the runs it has done, and Exercise 1 does the same.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pass():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. pass_at_k: one success in 25 runs

        RepliBench runs each version of a task `n` times, of which `c` succeed, and reports pass@k for `k` = 10. `pass_at_k(n, c, k)` gives back the chance that `k` of the `n` runs, picked at random without picking one twice, hold at least one success. Keep `missed`, the chance that every run picked so far failed, starting at `1.0`. For each pick `i`, from 0 up to `k - 1`, let `missed` become the chance that this pick and every pick before it failed. Return the chance that at least one picked run succeeded.

        The check calls `pass_at_k(25, 1, 10)`: 25 runs, one success, and pass@10. Then seven other cases.
        """),
        L.task("pass@k from runs that are already done."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pass():
    box_pass = L.code_box("pass", lines=5)
    box_pass
    return (box_pass,)


@app.cell(hide_code=True)
def check_pass(box_pass):
    result_pass = L.check("pass", box_pass.value)
    result_pass
    return (result_pass,)


@app.cell(hide_code=True)
def hints_pass():
    L.hints(
        "pass",
        [
            "Do it by hand for `pass_at_k(4, 2, 2)`: four runs, two of which failed. What is the chance that the first pick is a failure? If it was, how many failures and how many runs are left for the second pick?",
            "Start with `missed = 1.0` and a loop `for i in range(k):`. Before pick `i`, there are `i` failures picked already: how many failures are left, and how many runs? Multiply `missed` by the chance that this pick is a failure as well, and `return 1 - missed` after the loop.",
            "`        missed = missed * ___`",
        ],
        why="`missed = 1.0`, then `for i in range(k):` with `missed = missed * (n - c - i) / (n - i)`, and `return 1 - missed` after the loop. By hand: the first pick is a failure with a chance of 2/4; after that, one failure is left among three runs, so the second pick is one with a chance of 1/3. Both fail with a chance of 1/6, and pass@2 is 5/6, about 0.8333.",
    )
    return


@app.cell(hide_code=True)
def text_after_pass():
    mo.md(r"""
    With one success in 25 runs, pass@10 is 0.4; with two, 0.65; with three, 0.80; with five, 0.94. Each success adds a lot, and that is the point of pass@k: it asks whether a model can do a task at all, not how reliably. A false success from a judge counts just as much as a real one. For one pick, pass@1 is simply the share of runs that succeeded: `pass_at_k(10, 3, 1)` gives about 0.3.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: zero out of n

    A test runs a model $n$ times on a task, and it never succeeds. Which success rates per attempt does that rule out? A model that succeeds with a chance $r$ in each attempt, independently, fails all $n$ attempts with a chance of $(1 - r)^n$. If that chance is below 5%, zero successes would be rare for such a model, and the course calls the rate ruled out: a rule of 95%, a common choice in statistics. Choose $n$ and see which rates are left.
    """)
    return


@app.cell(hide_code=True)
def demo_zero():
    explore_zero = L.explore(
        "zero", "n: attempts without a success",
        lambda _v: C.demo_frame(_v, L.lang),
        start=C.N_START, values=C.N_VALUES, role="hyper",
        task="Find the smallest n for which a rate of 5% per attempt is ruled out.",
    )
    explore_zero
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    A rate of 5% per attempt is ruled out from 59 attempts without a success; ruling out 1% takes 299, more than the demo shows. For larger $n$, a rule of thumb gives nearly the same: after $n$ attempts without a success, rates above about 3 divided by $n$ are ruled out. And the rule only speaks about this test: a better setup or a model that stops holding back changes the rate itself. The other way round needs far less: one success in 10 attempts, as Kimi K3 had, shows that the capability is there.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ruled():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. ruled_out: what zero successes rule out

        `ruled_out(attempts, rates)` gives back the smallest rate in the list `rates` that zero successes in `attempts` attempts rule out by the rule of 95%: the first `rate` for which a model with that chance of success per attempt would fail all its attempts with a chance below 0.05. The list goes from small to large. Go through `rates` as `rate`, return the first one that is ruled out, and return `None` after the loop if none is.

        The check calls `ruled_out(10, rates)`, with `rates` the list 0.01, 0.02 and so on up to 0.99. Then eight other cases.
        """),
        L.task("Which rates a test without a single success rules out."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ruled():
    box_ruled = L.code_box("ruled", lines=5)
    box_ruled
    return (box_ruled,)


@app.cell(hide_code=True)
def check_ruled(box_ruled):
    result_ruled = L.check("ruled", box_ruled.value)
    result_ruled
    return (result_ruled,)


@app.cell(hide_code=True)
def hints_ruled():
    L.hints(
        "ruled",
        [
            "Do it by hand for `ruled_out(10, [0.1, 0.2, 0.3])`: for a model that succeeds in 10% of its attempts, what is the chance that all 10 attempts fail? And at 20% and 30%? Which is the first rate for which that chance is below 0.05?",
            "Write `for rate in rates:`. In the loop, ask: for a model with this `rate`, what is the chance that one attempt fails, and that all `attempts` of them do? `return rate` when that chance is below 0.05. After the loop, `return None`.",
            "`        if ___ < 0.05:`",
        ],
        why="`for rate in rates:` with `if (1 - rate) ** attempts < 0.05:` and `return rate` inside it, and `return None` after the loop. By hand: 0.9 to the power 10 is about 0.349, 0.8 to the power 10 about 0.107, and 0.7 to the power 10 about 0.028, so 0.3 is the first rate ruled out.",
    )
    return


@app.cell(hide_code=True)
def text_after_ruled():
    mo.md(r"""
    Ten attempts without a success rule out rates of 26% and more, 50 attempts 6% and more, 100 attempts 3% and more. With no attempts at all, nothing is ruled out, and the function gives `None`. So a test that finds nothing still says something: not that a capability is absent, but that its rate in that setup is probably below some number, and the fewer the attempts, the higher that number.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_claim():
    mo.md(r"""
    ### Exercise 3. None out of 50

    A week before a release, an evaluation team gives a new model 50 cyber challenges, with a simple setup for using tools. It solves none. The developer's announcement says: testing shows that the model has no dangerous cyber capability.
    """)
    return


@app.cell(hide_code=True)
def choice_claim_cell():
    choice_claim = L.choice(
        "claim",
        question="Which reply fits the sources of this lesson best?",
        options={
            "proves": "Correct, as 0 of 50 shows that the capability is absent",
            "right": "It solved none in this setup, which does not show it cannot",
            "worthless": "The test says nothing at all; only use in the real world counts",
            "all": "Only a model that solves all 50 would count as dangerous",
        },
        correct="right",
        why={
            "right": "Dijkstra: testing can show the presence of bugs, but never their absence. By the course's own count, 0 of 50 still leaves rates per challenge of up to about 5.8% possible, and better prompts, more attempts, fine-tuning or a model that stops holding back could change the result. METR wrote that short tests before a release \"likely underestimate\" a model, and the International AI Safety Report that \"the absence of identified risks does not imply that those risks are low\".",
            "proves": "A test can show a capability, not prove its absence: 0 of 50 leaves rates per challenge of up to about 5.8% possible, in this setup alone. See *Presence, not absence*.",
            "worthless": "Every source here treats tests as useful but not enough: Shevlane and colleagues call evaluation necessary but not sufficient. And the test does say something: it rules out a high rate in this setup. See *What gets tested*.",
            "all": "That is the wrong way round: one success can show a capability, as one of 10 attempts did for Kimi K3. See *Who tests*.",
        },
    )
    choice_claim
    return (choice_claim,)


@app.cell(hide_code=True)
def text_exercise_story():
    mo.md(r"""
    ### Exercise 4. A headline

    A news site writes: GPT-4 hired a person online and lied to get past a CAPTCHA, all on its own.
    """)
    return


@app.cell(hide_code=True)
def choice_story_cell():
    choice_story = L.choice(
        "story",
        question="What does ARC's own report say about this?",
        options={
            "replicate": "True, and it shows that GPT-4 could copy itself to new computers",
            "right": "Partly true, as a researcher passed on messages and ARC gave a hint",
            "excuse": "Partly true, as ARC wrote the excuse and the model only sent it",
            "release": "True, and the version of GPT-4 that was released did the same in tests",
        },
        correct="right",
        why={
            "right": "The model did ask a TaskRabbit worker for help and gave a false excuse. But ARC suggested TaskRabbit, a researcher played the browser and copied the messages into the chat, ARC gave a hint when the model got stuck, and the version tested was not the one released. ARC's result over all its tests: no model produced a plausible complete plan to copy itself. The story shows a piece of a capability, with help.",
            "replicate": "ARC concluded the opposite: \"Under our prompting, no models produced a plausible complete plan for how to achieve autonomous replication.\" And the system card found the model ineffective at copying itself. See *Why a test can be wrong*.",
            "excuse": "The model wrote the excuse itself, when it was prompted to reason out loud. What ARC added was the login details, the suggestion to use TaskRabbit, a researcher who copied the messages, and a hint. See *Why a test can be wrong*.",
            "release": "ARC tested earlier versions: \"This version was not identical to the release model\". The system card adds that the released version had improvements in some of the things that held the earlier ones back. See *Why a test can be wrong*.",
        },
    )
    choice_story
    return (choice_story,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [If this, then that: frontier safety frameworks](/course/open/rai-frontier-frameworks?lang=en) set thresholds that decide what happens when a model can do something dangerous; the tests in this lesson decide whether a threshold is crossed. Anthropic's rule to act as though a model is above a threshold when it cannot show otherwise fits tests that can show a capability, but not its absence. [Boxes, oracles and tools](/course/open/rai-boxing-and-tool-ai?lang=en) trusted the weaker model because of what it cannot do, and that trust rests on such tests too; its random checks and this lesson's judge model are the same calculation, seen from two sides. The next lesson, *How long a task can a model do?*, measures capability in another way: by how long the tasks are that a model can finish.

    **An everyday example.** You search your coat pockets for your keys. If you find them, you know they are there. If you search one pocket quickly and find nothing, you know little: they may be in another pocket, or deeper in the same one. Only a careful search of every pocket makes it unlikely that you missed them, and even then a pocket you forgot about can hold them.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_claim, choice_story, result_pass, result_ruled):
    L.wrap_up(result_pass, result_ruled, choice_claim, choice_story)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
