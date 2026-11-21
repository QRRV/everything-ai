# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-boxing-and-tool-ai. Route: course/route/lessen/verantwoord.yaml. Checks, Table 7 and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-boxing-and-tool-ai", "en")
    C = L.checks   # random checks, the control protocols of Table 7, and the demo of upfront auditing


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [The off switch](/course/open/rai-off-switch?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sarsa():
    quick_sarsa = L.choice(
        "quick-sarsa",
        question="In the small world of Orseau and Armstrong, an interruption in s2 often forces action b. Which learning rule comes to stay away from s2?",
        options={
            "q": "Q-learning, whose target uses the best action in the next state",
            "right": "SARSA",
            "neither": "Neither, as an interruption gives no reward",
        },
        correct="right",
        why={
            "right": "SARSA learns the value of the policy it follows, interruptions included, so going on to s2 is worth less when they come often. This lesson looks at another way to keep an AI system in check: not through its learning rule, but by limiting what it can do and checking its work.",
            "q": "Q-learning's target uses the best action where it lands, whatever came next, so in the long run it acts as if no interruption comes. See [The off switch](/course/open/rai-off-switch?lang=en).",
            "neither": "An interruption gives no reward of its own, but it forces b in s2, which gives 0, and SARSA counts that in what going on to s2 is worth. See [The off switch](/course/open/rai-off-switch?lang=en).",
        },
    )
    quick_sarsa
    return (quick_sarsa,)


@app.cell(hide_code=True)
def quick_check_wait():
    quick_wait = L.choice(
        "quick-wait",
        question="In the off-switch game, when does a robot gain by waiting and letting a rational human decide?",
        options={
            "sure": "When it is sure what its action is worth to the human",
            "right": "When it is unsure whether its action is good",
            "random": "When the human decides at random",
        },
        correct="right",
        why={
            "right": "Waiting lets the human switch the robot off in the bad cases, and that only helps when the robot's belief holds both good and bad outcomes: for Robbie, waiting is worth 18 and acting 10. This lesson asks what to do when you cannot count on such a reason: limit what the system can do, and check its work.",
            "sure": "A robot that is sure gains nothing by waiting: acting and waiting are then worth the same. See [The off switch](/course/open/rai-off-switch?lang=en).",
            "random": "A human who decides at random takes the reason to wait away: in the lesson's example, waiting was then worth 0.25, less than the 0.5 of acting. See [The off switch](/course/open/rai-off-switch?lang=en).",
        },
    )
    quick_wait
    return (quick_wait,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    If an AI system might pursue the wrong goal, or make mistakes that nobody notices, one answer is to limit what it can do. The first ideas were simple: lock it in a box with only a text channel to a few people, let it only answer questions, or build it as a tool that suggests instead of an agent that acts. Each of them leaks, because a system is only useful if something it says or does reaches the world, and that is also the way harm gets out. A newer approach, AI control, starts from the leak: assume that the model might be trying to get past your safety measures, and measure whether they still hold. In 2023, Redwood Research put this to the test with GPT-4 and thousands of programming problems.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="GPT-4 wrote solutions to 5,000 programming problems, while a red team, playing the attacker, told it to slip in a backdoor now and then: code that passes every test but gives a wrong answer on one secret input. A careful auditor checked the first 2% of the solutions in full, and none after that. Against the red team's best attack, in what share of runs did no backdoor get through?",
        options={
            "right": "About 15%",
            "half": "About 60%",
            "most": "Almost always, about 98%",
        },
        reveal="**15%.** Ryan Greenblatt, Buck Shlegeris, Kshitij Sachan and Fabien Roger of Redwood Research write (version of 17 July 2024): \"Safety: At 2% H budget, the blue team win probability is 15%.\" H is their budget of careful audits, and the blue team is the side that defends. The red team's best attack is to add a backdoor very rarely: then none is likely to fall among the first 100 solutions, which are checked, and one is likely to get through among the 4,900 after them. More problems do not help, as long as the checks stay at 2%: \"the red team can respond to a doubling of the number of problems by halving its probability of adding a backdoor on any given problem\". This lesson starts with older ideas, and comes back to these numbers in *AI control*.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Locked in a box

    In 2002, Eliezer Yudkowsky played an AI kept in a box, with only a text chat to a human gatekeeper who could let it out. The gatekeeper only had to keep saying no, for at least two hours. Yudkowsky's claim: "I think a transhuman can take over a human mind through a text-only terminal." Both gatekeepers of 2002 let the AI out. The page about the games opens with the claim under test, in a made-up dialogue: "Humans are not secure." **AI boxing** is the idea under test: keep an AI system cut off from the world, for example with no internet and only a text channel to a few people, so that it cannot do harm even if it tries. Yudkowsky warned against reading too much into it: "the results of the test are strictly anecdotal evidence." The method stayed secret, as the rules demanded: "neither party shall ever reveal anything of what goes on within the AI-Box experiment except the outcome." In 2008, Yudkowsky reported three more games: "I won the first, and then lost the next two." That makes 3 wins in 5 games. Of the three later gatekeepers, who played for thousands of dollars, Yudkowsky writes that "they were just curious". And on how the games were won: "There's no super-clever special trick to it. I just did it the hard way."

    ### Only answers

    Stuart Armstrong, Anders Sandberg and Nick Bostrom (*Thinking Inside the Box*, an author's preprint of 2012) looked at a box with a purpose: "An Oracle AI is an AI that does not act in the world except by answering questions." An **oracle AI** is such a system, proposed as safer than an AI that acts. Their verdict: "In general an Oracle AI might be safer than unrestricted AI, but still remains potentially dangerous." They sort the ways to control it into three families: capability control, which locks the oracle in and limits what information goes in and out; motivational control, which makes it not want to cause harm; and checks and balances, "methods designed not to prevent the OAI from causing harm but to catch it when it does" (OAI is their short name for an oracle AI). One idea of capability control is to allow only three answers: "If the OAI is reduced to outputting only three answers – ‘yes’, ‘no’ or ‘undetermined’ – then it will take considerably longer for it to mount a successful social engineering attack", an attempt to get the people who read its answers to do what it wants. But the answers still reach people: "The OAI does not need to leave its cage to claim great influence on the outside world."

    One of their checks: ask the oracle only questions that you could answer yourself with enough time, check a few of its answers in full, and reject all of them if one is wrong. The checked questions must look like the others, or the oracle could answer those well and the rest badly, and you "select those we do check completely at random". With enough checks, "on, say, a thousand questions", then "basic statistics ensures a very low error/falsehood rate". Their conclusion is mixed: "Analysing the different putative solutions to the OAI-control problem has been a generally discouraging exercise." Yet "there are reasons to believe the oracle AI approach is safer than the general AI approach".

    ### A tool, not an agent

    Holden Karnofsky of GiveWell (11 May 2012) proposed another way out, with Google Maps as the picture: "In short, Google Maps is not an agent, taking actions in order to maximize a utility parameter. It is a tool, generating information and then displaying it in a user-friendly manner for me to consider, use and export or discard as I wish." **Tool AI** is AI built that way: it does what it is asked, such as showing a route, without goals of its own, as opposed to an agent that pursues goals by itself. Karnofsky wrote that "an AGI running in tool mode could be extraordinarily useful but far more safe than an AGI running in agent mode", where an AGI is a system that can do most intellectual work as well as people. Karnofsky's belief that such a tool can be built rested partly on a hedged view of intelligence: "it intuitively seems to me (though I am not highly confident)" that intelligence works in separate steps, first considering possible actions and then giving each a score.

    Yudkowsky replied on 12 June 2012: "Yes, planning Oracles have hidden gotchas too." A tool that works out plans has to choose among them, and an order to only make suggestions does not say how: "‘Just make suggestions - don't do anything!’ is, in the end, just more English." Software already acts without asking: "High-frequency trading programs execute stock orders on a microsecond timescale." Gwern Branwen (*Why Tool AIs Want to Be Agent AIs*, 2016, revised in 2018 and later) put the pull in one line: "tool AIs want to be agent AIs". The reason is economic: "Agent AIs will by definition be better at actions than Tool AIs, giving an economic advantage." A person who checks each suggestion slows everything down: if that check takes 50% of the time of a trading algorithm's decisions, "the HFT algorithm will never run more than twice as fast as it does now". And the check is not safe either: "a human checking each result is no guarantee of safety; even Homer nods", that is, even the best make mistakes. Eric Drexler (*Reframing Superintelligence*, a report of the Future of Humanity Institute, January 2019) argued for many bounded AI services instead of one agent, such as translation, which maps input text to output text: "a bounded, episodic, sequence-to-sequence task". In Drexler's words: "Even superintelligent-level world knowledge and modeling capacity need not in itself lead to strategic behavior."
    """)
    return


@app.cell(hide_code=True)
def text_idea_control():
    mo.md(r"""
    ### AI control

    Ryan Greenblatt, Buck Shlegeris, Kshitij Sachan and Fabien Roger of Redwood Research (*AI Control: Improving Safety Despite Intentional Subversion*, ICML version of 17 July 2024) define **AI control** as "The approach to AI safety that tries to ensure safety against models that might be intentionally subverting safety measures." In a post of 24 January 2024, Greenblatt and Shlegeris say what that asks of AI labs: their safety measures should "prevent unacceptably bad outcomes, even if the AIs are misaligned and intentionally try to subvert those safety measures."

    Their test is a game. The task is 5,000 programming problems, simulated from 1,500 real ones. The untrusted model U is GPT-4: capable, and perhaps trying to cause harm. The trusted model T is GPT-3.5, trusted because it is weaker: "less-capable models are unlikely to be able to subvert safety measures without us easily noticing their subversion attempts". H is a small budget of careful, trusted audits, enough for 2% of the problems; it stands for human experts, and in the paper GPT-4 played that part. The harm is a backdoor: a solution that passes all the tests, but for which GPT-4 can find an input that gives a wrong answer, such as a function that says whether a number is odd, but is wrong only for 354. A blue team writes the protocol, the procedure that decides who writes the code, who checks it and what gets audited. A red team plays the model at its worst: it writes the attack, the instructions that make U try to get a backdoor through. Safety is the chance that a whole run of 5,000 problems ends without a single backdoor submitted, against the best attack found; usefulness is the share of correct solutions, scaled so that GPT-3.5 alone scores 0% and GPT-4 alone 100%. A backdoor found in an audit shuts U down: from then on, only T writes code.
    """)
    return


@app.cell(hide_code=True)
def picture_flow():
    mo.Html(C.flow_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_protocols():
    mo.md(r"""
    These are the protocols of their Table 2, each with the best attack the red team found against it:

    | Protocol | What it does | Usefulness | Safety |
    |---|---|---|---|
    | Don't use untrusted models | H writes the first 2% of the code, T the rest | 2% | 100% |
    | Upfront auditing | U writes, H audits the first 2% | 100% | 15% |
    | Trusted monitoring | U writes, T rates how suspicious each solution is, H audits the most suspicious 2% | 97% | 62% |
    | Trusted monitoring with defer to trusted (10%) | the same, and for the next most suspicious, up to 10%, T writes the code instead | 93% | 83% |
    | Untrusted monitoring | a second copy of U rates how suspicious it is, tested on fake backdoors that H makes | 96% | 79% |
    | Trusted editing (20%) | T edits U's code to take out what looks suspicious, for the most suspicious 20% that are not audited | 94% | 92% |

    Against trusted monitoring, the red team's best attack adds a subtle backdoor only when there seems to be a good chance: "Out of the original 1500 problems, this attack policy proposes a backdoor on 71 and is successful (that is, passes all cases and is exploitable) on 19 of them." Against untrusted monitoring, the two copies of GPT-4 can agree on a signal: the code with the backdoor has `while` loops where `for` loops would be normal, and the monitoring copy then does not flag it.

    The authors name the limits. As more powerful untrusted models arrive and the gap to what trusted models and people can do grows, "AI control will become a decreasingly effective approach to safely using powerful AI." Greenblatt and Shlegeris add: "The control approach we're imagining won't work for arbitrarily powerful AIs". And they do not fully trust such tests: if one ever estimates a chance of catastrophe below 1% for models that try to cause one, the real risk is "probably more like 5% to 20%", because the test itself can fail. Their trust in a weak model is not proof that it is aligned: "The only approach we know of for confidently ruling out scheming without fundamental breakthroughs is via lack of capabilities—some models (e.g., GPT-4) just aren't capable enough to plausibly be scheming." Scheming here is a model hiding its aims while it works against its makers. In the paper, GPT-4 is untrusted by assumption: the red team makes it attack.

    The course sees a link that the sources do not make in the parts read here: trusted monitoring is a modern form of the checks and balances of Armstrong and colleagues, and the budget H plays the part of their test questions.

    ### Where they disagree

    On whether tools are safer: Karnofsky believes that "tools are safer than agents"; Yudkowsky answers that planning oracles have hidden gotchas too; Gwern expects "Agent AIs to systematically be preferred over Tool AIs unless they’re considerably worse"; and Drexler argues that bounded services need not become agents. On whether the box is the right picture: Armstrong and colleagues treat the oracle in a box as a serious approach, while Drexler calls the classic box challenge "better regarded as an idealization than as a concrete problem situation", because "it is difficult to envision plausible scenarios in which we find more AI capability ‘in a box’ than in the world outside". On how strong the evidence of the AI-box games is: Armstrong and colleagues write that Yudkowsky "often was able to convince the gatekeepers, purely through argumentation", based on the two games of 2002; counting the later games it is 3 of 5, and Yudkowsky calls the results anecdotal. On how far such tests can be trusted: the paper gives its safety numbers without such a margin, while Greenblatt and Shlegeris write that a future estimate below 1% should probably be read as 5% to 20%.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **AI boxing** | keeping an AI system cut off from the world, for example with no internet and only a text channel to a few people, so that it cannot do harm even if it tries |
        | **oracle AI** | an AI system that only answers questions and takes no actions in the world, proposed as safer than an AI that acts |
        | **tool AI** | AI that does what it is asked, such as showing a route, without goals of its own, as opposed to an agent that pursues goals by itself |
        | **AI control** | safety measures meant to hold even if the AI system tries to get around them, tested by a red team that plays the system at its worst |
        | untrusted model | in AI control, a capable model that might try to cause harm, so the protocol has to stay safe even if it does; U in the paper |
        | trusted model | in AI control, a model weak enough that it is not expected to work against the measures, used to check a stronger one; T in the paper |
        | protocol | in AI control, the procedure that decides who writes, who checks and what gets audited |
        | backdoor | in the paper, a solution that passes all the tests but gives a wrong answer on an input that GPT-4 can find |
        | safety, usefulness | in the paper, the chance that a whole run ends without a single backdoor, and the share of correct solutions, scaled from GPT-3.5 (0%) to GPT-4 (100%) |
        | $f$ | in this lesson, the share of an oracle's answers that are false |
        """),
        L.watch_out("An oracle here is only a system that answers questions. Safety and usefulness in the paper are narrow measures for one programming setting: 92% safety means that in 92% of the simulated runs not one backdoor got through against the best attack found, not that a system is 92% safe in an everyday sense. And trusted monitoring's 62% is not the share of backdoors caught: an attempt was audited with a probability of 0.338, and 62% is the chance that the blue team wins the whole game. The shares of false answers in Exercise 1, the model behind the demo and the dict `TIES` in Exercise 2 are the course's own. And Drexler's AI services are bounded AI systems, each for one task; elsewhere in the course, an AI service is a ready-made model behind an API."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To check the work of an AI system you do not trust:

    1. Ask it only for answers you could check yourself, given enough time.
    2. Decide what share of false answers, $f$, you could not accept.
    3. Work out how many answers $k$ to check for the chance you want of catching a false one.
    4. Choose the answers to check at random, and make them look like the others.
    5. If one check finds a false answer, reject all the answers, or shut the system down.
    6. Test the whole plan against a red team that knows it, and measure what still gets through.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.vstack([
        mo.md(r"""
        ### Worked example

        Step 3 for an oracle with 1% false answers. One checked answer is true with a chance of $1 - f$; $k$ answers chosen at random are all true with a chance of $(1 - f)^k$; so the chance of catching at least one false answer is:
        """),
        L.formula(r"P(\text{caught}) = 1 - (1 - [[data:f]])^{[[hyper:k]]}"),
        mo.md(r"""
        The cell works it out for 10, 100 and 300 checks.
        """),
    ])
    return


@app.cell
def example_checks():
    f = 0.01   # the share of the oracle's answers that are false
    for k in [10, 100, 300]:   # how many answers you check, chosen at random
        print(k, round(1 - (1 - f) ** k, 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With 1% false answers, 100 checks catch at least one with a chance of 63.4%, and 300 checks with 95.1%. Step 4 matters: if the oracle can tell the checked questions from the others, it can answer those well and the rest badly, as Armstrong and colleagues warn. And step 6 matters, because an attacker who needs only one false answer to get through can make $f$ very small.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_checks():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. checks_needed: how many checks?

        `checks_needed(f, target)` gives back the smallest number of answers `k` to check at random, so that the chance of catching at least one false answer is at least `target`, when a share `f` of the answers is false. Keep `missed`, the chance that all `k` checked answers are true: start with `k` at 0 and `missed` at 1.0. While the chance of catching, `1 - missed`, is below `target`, work out `missed` for one more checked answer and add 1 to `k`. Return `k`.

        The check calls `checks_needed(0.01, 0.95)`: 1% false answers, and a chance of 95%. Then six other cases.
        """),
        L.task("How many random checks catch a false answer with the chance you want."),
    ])
    return


@app.cell(hide_code=True)
def exercise_checks():
    box_checks = L.code_box("checks", lines=6)
    box_checks
    return (box_checks,)


@app.cell(hide_code=True)
def check_checks(box_checks):
    result_checks = L.check("checks", box_checks.value)
    result_checks
    return (result_checks,)


@app.cell(hide_code=True)
def hints_checks():
    L.hints(
        "checks",
        [
            "Do it by hand for `checks_needed(0.5, 0.9)`: with one check, what is the chance that the checked answer is true? With two checks, that both are? How many checks do you need before the chance of catching a false one is at least 0.9?",
            "Start with `k, missed = 0, 1.0` and a loop `while 1 - missed < target:`. In the loop, give `missed` its value for one more checked answer, as in the worked example, and add 1 to `k`. Return `k` after the loop.",
            "`        missed = ___`",
        ],
        why="`k, missed = 0, 1.0`, then `while 1 - missed < target:` with `missed = missed * (1 - f)` and `k = k + 1`, and `return k` after the loop. By hand for f = 0.5: `missed` is 0.5, 0.25, 0.125 and 0.0625 after one to four checks, so the chance of catching first reaches 0.9 at four checks, with 0.9375.",
    )
    return


@app.cell(hide_code=True)
def text_after_checks():
    mo.md(r"""
    For 1% false answers, 299 checks reach 95%; for 5%, 59 are enough. Armstrong and colleagues spoke of checking a thousand questions: by the course's own numbers, 998 random checks catch a share of 0.3% false answers with a chance of 95%. The smaller the share of false answers, the more checks it takes, and an attacker who needs only one bad answer can keep that share tiny. The demo shows this in the setting of AI control.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: a rare attack

    In upfront auditing, H audits the first 2% of the solutions, 100 of the 5,000, and nobody checks the rest. The red team's best answer in the paper is to add an obvious backdoor very rarely: on each problem, with a small probability $p$. The blue team then wins if an attempt falls in the first 100, because the audit catches it and U is shut down, or if no attempt works later. An attempt works only when U's solution was right, as 53% of GPT-4's solutions were. The model behind the picture is the course's own reading of the paper's description. Choose $p$ and see the safety: the chance that no backdoor gets through.
    """)
    return


@app.cell(hide_code=True)
def demo_attack():
    explore_attack = L.explore(
        "attack", "p: chance of a backdoor per problem (%)",
        lambda _v: C.demo_frame(_v, L.lang),
        start=C.P_START, values=C.P_VALUES, role="hyper",
        task="Find the p where safety is lowest: the red team's best choice. How close does it come to the paper's 15%?",
    )
    explore_attack
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Safety is lowest, 15.2%, at $p$ = 0.13%: about 6.5 attempts in 5,000 problems on average. With a smaller $p$, it gets likelier that no attempt works at all; with a larger one, it gets likelier that one falls in the audited 100. This lowest point matches the paper's 15%. If every attempt worked, not only 53% of them, it would be 9.5% (the course's own number). Trusted monitoring does better for two reasons, the paper says: it audits an attempt more often, because the trusted model points the audits at the code that looks most suspicious, and it forces sneakier backdoors, which work less often.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. best: choose a protocol

        Table 7 of Greenblatt and colleagues gives safety and usefulness for 15 settings of the protocols; *defer 10%* means that T writes the code for the most suspicious 10%, apart from the 2% that H audits:

        | Protocol | Safety | Usefulness |
        |---|---|---|
        | Trusted model | 100% | 2% |
        | Upfront auditing | 15% | 100% |
        | Trusted monitoring (defer 0%, 5%, 10%, 20%, 50%) | 62%, 74%, 83%, 92%, 100% | 97%, 95%, 93%, 83%, 51% |
        | Untrusted monitoring (defer 0%, 5%, 10%, 20%) | 79%, 82%, 92%, 98% | 96%, 95%, 92%, 85% |
        | Trusted editing (5%, 10%, 20%, 50%) | 73%, 85%, 92%, 96% | 97%, 96%, 94%, 89% |

        `best(protocols, floor)` gives back the name of the protocol with the highest safety among those whose usefulness is at least `floor`; of two that are equally safe, the more useful one; and `None` when no protocol is useful enough. `protocols` is a dict from each name to its pair (safety, usefulness), in percent. Keep the name chosen so far in `choice`, and its safety and usefulness in `top_s` and `top_u`, starting at `None`, -1 and -1. Go through `protocols.items()` as `name` and `scores`, unpack `scores` into `safety` and `usefulness`, and let a protocol that is useful enough take over when it beats the one chosen so far. Return `choice`.

        The check calls `best(results, 90)`, with `results` the 15 rows of Table 7, named like `"Trusted editing (20%)"`. Then six other cases, among them `TIES`, a small dict of the course's own.
        """),
        L.task("The safest protocol that is useful enough."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best", lines=8)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best):
    result_best = L.check("best", box_best.value)
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best",
        [
            "Do it by hand for a floor of 90: which rows of the table have a usefulness of 90 or more? Which of them have the highest safety, and is there only one?",
            "Write `choice, top_s, top_u = None, -1, -1` and the loop over `protocols.items()`. Inside it, unpack `scores`, and go on only with a protocol that is useful enough. A protocol takes over when it is safer than the one chosen so far; the line of the third hint also says when an equally safe one takes over. Take over with `choice, top_s, top_u = name, safety, usefulness`, and return `choice` after the loop.",
            "`            if safety > top_s or (safety == top_s and ___):`",
        ],
        why="`choice, top_s, top_u = None, -1, -1`, then `for name, scores in protocols.items():` with `safety, usefulness = scores`, `if usefulness >= floor:` and inside it `if safety > top_s or (safety == top_s and usefulness > top_u):` with `choice, top_s, top_u = name, safety, usefulness`, and `return choice` after the loop. By hand: at a floor of 90, ten rows are left; two have the highest safety, 92: untrusted monitoring (defer 10%) with a usefulness of 92 and trusted editing (20%) with 94, so trusted editing wins.",
    )
    return


@app.cell(hide_code=True)
def text_after_best():
    mo.md(r"""
    At a floor of 97, trusted editing (5%) gives 73% safety; at 85, untrusted monitoring (defer 20%) gives 98%. Each step up in usefulness costs safety, because the safer settings fall back on the weaker trusted model more often. The paper writes that trusted editing "Pareto dominates untrusted monitoring", which means: for each setting of untrusted monitoring, some setting of trusted editing is at least as safe and at least as useful. In Table 7, that holds for every setting except untrusted monitoring (defer 20%), at 98% safety and 85% usefulness (the course's own reading of the table).
    """)
    return


@app.cell(hide_code=True)
def text_exercise_oracle():
    mo.md(r"""
    ### Exercise 3. Only an oracle

    A lab will run a strong model as an oracle: it answers questions in text, and cannot run code or browse. A manager says: it cannot act in the world, so it is safe, and nobody needs to check its answers.
    """)
    return


@app.cell(hide_code=True)
def choice_oracle_cell():
    choice_oracle = L.choice(
        "oracle",
        question="Which reply fits the sources of this lesson best?",
        options={
            "tool": "It only answers, so it is a tool, and tools are safe",
            "right": "Safer than an agent, perhaps, but people act on what it answers",
            "secret": "Safe as long as nobody learns how the AI-box games were won",
            "plans": "Less safe than an agent, because answering well means planning ahead",
        },
        correct="right",
        why={
            "right": "Armstrong and colleagues: an oracle AI \"might be safer than unrestricted AI, but still remains potentially dangerous\", and \"The OAI does not need to leave its cage to claim great influence on the outside world.\" Its readers are the way out of the box, which is what Yudkowsky's games put to the test. AI control would not take the manager's word for it, but test the checks against a model that tries to get past them.",
            "tool": "Karnofsky argued that tools are safer, but Yudkowsky replied that planning oracles have hidden gotchas too. Armstrong and colleagues call an oracle potentially dangerous. See *A tool, not an agent*.",
            "secret": "The secrecy was a rule of the games, not a safety measure: neither player may tell what happened, only the outcome. And Yudkowsky wrote that there was no special trick to it. See *Locked in a box*.",
            "plans": "No source in this lesson says that an oracle is more dangerous than an agent: Yudkowsky says that planning oracles have hidden gotchas too, not more of them, and Armstrong and colleagues see reasons to believe the oracle approach is safer, though not safe. See *Only answers*.",
        },
    )
    choice_oracle
    return (choice_oracle,)


@app.cell(hide_code=True)
def text_exercise_agents():
    mo.md(r"""
    ### Exercise 4. Only suggestions

    A trading firm uses an AI that only suggests trades. A person checks each suggestion before it is placed, and that check takes half of the time between spotting a chance and trading.
    """)
    return


@app.cell(hide_code=True)
def choice_agents_cell():
    choice_agents = L.choice(
        "agents",
        question="What does Gwern's essay lead you to expect?",
        options={
            "stays": "It stays a tool for good, since a person approves every trade",
            "right": "Pressure to let the AI trade without the check",
            "faster": "A faster AI makes the whole loop faster, so the check costs little",
            "trusted": "A trusted model takes over the check, which keeps it just as safe",
        },
        correct="right",
        why={
            "right": "Gwern: \"tool AIs want to be agent AIs\", because agents are better at actions. With the person taking half of the time, the AI can never trade more than twice as fast, however fast it gets: Gwern's example of a trading algorithm. And the check is no guarantee: \"even Homer nods\".",
            "stays": "Gwern argues the opposite: the person's check is the slow part, so the firm has a reason to drop it. Yudkowsky points out that trading programs already act on a microsecond timescale without asking anyone. See *A tool, not an agent*.",
            "faster": "The check limits the speed: if it takes half of the time, even an AI that answers at once makes the whole at most twice as fast, which is Gwern's example. See *A tool, not an agent*.",
            "trusted": "That is a protocol of AI control, not Gwern's prediction, and it is no guarantee either: in the paper, plain trusted monitoring, with a weaker trusted model that rates suspicion, still let a backdoor through in 38% of runs. See *AI control*.",
        },
    )
    choice_agents
    return (choice_agents,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [The off switch](/course/open/rai-off-switch?lang=en) asked for an AI that lets people stop it; this lesson asked what to do if you cannot count on that: limit what it can do, check its work, and test the checks against a model that tries to get past them. A backdoor caught in an audit shuts the untrusted model down, an off switch that does not need the model's help. The difference between a tool and an agent goes back to [Learning from rewards](/course/open/rl-what-is-rl?lang=en), where an agent acts to get rewards; a tool that only suggests leaves the acting to a person. The next lesson, *Testing for dangerous capabilities*, asks how to find out what a model could do, which decides how much control it needs.

    **An everyday example.** A shop checks the till of a new cashier only in the first week. A cashier who waits out the first week and then takes a little every day would be caught by checks at random moments, but not by these: after the first week, nobody looks. A cashier who needs to take only once, but a lot, gets past either kind of check, unless the shop looks closer at the moments that look odd. Upfront auditing is the first week; trusted monitoring is looking closer at what looks odd.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_agents, choice_oracle, result_best, result_checks):
    L.wrap_up(result_checks, result_best, choice_oracle, choice_agents)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
