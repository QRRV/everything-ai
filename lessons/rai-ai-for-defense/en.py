# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-ai-for-defense. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow offence-defence balance

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-ai-for-defense", "en")
    C = L.checks   # a made-up piece of software with 10 vulnerabilities, after Garfinkel and Dafoe's model


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Races and selection](/course/open/rai-race-and-selection?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_worth():
    quick_worth = L.choice(
        "quick-worth",
        question="In Askell and colleagues' game, racing is worth 3 to a company if the rival stays careful, and 1 if the rival races. The company's trust that the rival stays careful is 0.5. What is racing worth to it on average?",
        options={
            "half": "1.5, half of 3",
            "right": "2",
            "sum": "4, the two worths added up",
        },
        correct="right",
        why={
            "right": "Weigh each worth by its chance and add them up: 0.5 × 3 + 0.5 × 1 = 2. That is the expected value of racing, as in Askell and colleagues' rule.",
            "half": "That leaves out what racing is worth when the rival races. Weigh both worths by their chances: 0.5 × 3 + 0.5 × 1 = 2. See [Races and selection](/course/open/rai-race-and-selection?lang=en).",
            "sum": "Only one of the two outcomes happens, so adding the worths counts too much. Weigh each by its chance: 0.5 × 3 + 0.5 × 1 = 2. See [Races and selection](/course/open/rai-race-and-selection?lang=en).",
        },
    )
    quick_worth
    return (quick_worth,)


@app.cell(hide_code=True)
def quick_check_kinds():
    quick_kinds = L.choice(
        "quick-kinds",
        question="Two kinds of AI agent start with the same number of copies. Each generation, the copies of one kind are multiplied by 1.1, those of the other by 1. What happens to the share of the first kind over many generations?",
        options={
            "half": "It stays at half",
            "right": "It keeps growing, toward the whole population",
            "once": "It grows once, to 1.1 times half, and then stays there",
        },
        correct="right",
        why={
            "right": "The first kind gains 10% each generation and the second gains nothing, so the first kind's share keeps rising toward 1, as the kind with 1.1 did in the made-up population of the last lesson.",
            "half": "Both kinds keep their copies, but the first gains 10% each generation and the second gains nothing, so the first kind's share keeps growing. See [Races and selection](/course/open/rai-race-and-selection?lang=en).",
            "once": "The multiplying happens every generation, not once: after two generations the first kind has 1.1 × 1.1 times its copies, and so on, so its share keeps growing. See [Races and selection](/course/open/rai-race-and-selection?lang=en).",
        },
    )
    quick_kinds
    return (quick_kinds,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The International AI Safety Report 2026, the second edition of a report written by experts under the chair of Yoshua Bengio and published in February 2026, writes: "There is now strong evidence that criminal groups and state-sponsored attackers actively use AI in their cyber operations." But according to the report, whether AI systems have increased the overall scale and severity of cyberattacks "remains uncertain because establishing causal effects is difficult." According to the report, one AI developer reported in November 2025 "that a threat actor used their models to automate 80–90% of the effort involved in an intrusion, with human involvement limited to critical decision points." The same report also writes: "Many of the same AI capabilities used for cyberattacks can also strengthen defences, creating uncertainty about whether AI benefits attackers or defenders more." In a blog post of 1 November 2024, a team at Google reported that its AI agent, Big Sleep, had found an exploitable vulnerability in SQLite, a widely used database engine, before it appeared in an official release. Does AI help those who attack more than those who defend? This lesson looks at AI used to make the systems around it safer, and at a simple model, which you program yourself, of how the balance between attack and defense can shift when tools get better.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In the final of DARPA's AI Cyber Challenge, whose results DARPA announced on 8 August 2025, teams let their AI systems search open-source software in which the organizers had put 63 vulnerabilities. How many of them did the systems find, all teams together?",
        options={
            "quarter": "About 1 in 4",
            "half": "About half of them",
            "right": "About 6 in 7",
        },
        reveal="**About 6 in 7.** According to DARPA, \"competitors’ systems discovered 54 unique synthetic vulnerabilities in the Final Competition’s 63 challenges. Of those, they patched 43.\" At the semifinal of August 2024, they found 37%, DARPA writes. The 63 vulnerabilities were made for the competition: they show what the systems did on this test, not how many vulnerabilities AI finds in all software.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Systemic safety

    In *Unsolved Problems in ML Safety* (version 5 of 16 June 2022), Dan Hendrycks, Nicholas Carlini, John Schulman and Jacob Steinhardt write: "ML systems are more likely to fail or be misdirected if the larger context in which they operate is insecure or turbulent." Their answer is a research area of its own: "Systemic safety research applies ML to mitigate potential contextual hazards that may decisively cause ML systems to fail or be misdirected." As two examples, they name research on cybersecurity and on informed decision making. Advances in code translation and code generation, they write, suggest that future models could apply security patches and make code more secure, "so that future systems not only flag security vulnerabilities but also fix them."

    Hendrycks's textbook *Introduction to AI Safety, Ethics, and Society* (a draft manuscript, version 2 of 15 November 2024) puts it this way: "AI can be used to help defend against the potential risks it poses." It names three examples: defenses against cyberattacks, security against pandemics, and a better information environment. Against cyberattacks, AI "could also be used to find vulnerabilities in codebases, shoring up defenses against AI-enabled hacking." Against pandemics, the textbook mentions "real-time epidemic detection" and "screening synthesis of gene sequences to prevent misuse". For information, AI could help society "by detecting false claims and providing evidence-backed answers to questions." According to the textbook, this approach has been called "systemic safety", while "the broader idea of focussing on technologies that defend against societal risks is sometimes described as ‘defensive accelerationism’". It also sets a limit: AI "is not the only solution, and investment in other areas such as physical infrastructure and equipment could be even more valuable", such as better ventilation and filtering of the air in buildings, which makes it harder for respiratory viruses to pass from one person to another.

    ### The offense-defense balance

    Ben Garfinkel and Allan Dafoe, of the Future of Humanity Institute at the University of Oxford, write in the *Journal of Strategic Studies* (2019): "Informally, the offense-defense balance refers to the relative ease of carrying out and defending against attacks." Such a balance is not one number for the whole world: "An offense-defense balance is defined relative to a particular objective, form of conflict, and set of actors." Their question is what happens when both sides invest more. According to their abstract, simple models of ground invasions and of cyberattacks "suggest that, in both cases, growth in investments will favor offense when investment levels are sufficiently low and favor defense when they are sufficiently high. We refer to this phenomenon as offensive-then-defensive scaling or OD-scaling."

    Their model of cyberattacks is about zero days: vulnerabilities that the defender does not know yet. An attacker acquires a zero day, they write, "when it discovers a vulnerability that the defender has not." They assume that the two sides' discoveries are not correlated, that no new vulnerabilities appear, and that each side finds vulnerabilities in step with its effort, until all are found. Take n vulnerabilities, and count effort so that one unit gives one expected discovery. The attacker then finds the smaller of n and its effort, and each vulnerability it finds is unknown to the defender with a chance equal to the share the defender has not found. The expected number of zero days is the product:

    $$\text{zero days} = \min(n, \text{attack}) \cdot \frac{n - \min(n, \text{defend})}{n}$$

    Why more effort first helps the attacker and then the defender, they explain like this: "Initially, due to randomness, the two actors are unlikely to discover precisely the same vulnerabilities." But the number of vulnerabilities is finite, so "the defender will eventually begin to achieve saturation by finding very large portions of them." They apply this to automated vulnerability discovery: if better tools count as more effort for both sides, then, all else being equal, "a plausible effect of this trend would be to tilt the relevant offense-defense balance further toward offense in the short-run, then toward defense in the long run." They call their models highly idealized, and they set a limit themselves: "We have shied away from drawing specific conclusions about the relationship between scaling effects and international security." OpenAI's Preparedness Framework (version 2 of April 2025) uses the same idea for cybersecurity, one of the areas the framework tracks: removing bottlenecks that limit malicious cyber activity "may upset the current cyberoffense-cyberdefense balance by significantly automating and scaling the volume of existing cyberattacks."

    ### Two defenders at work

    In that blog post of 1 November 2024, the Big Sleep team, of Google Project Zero and Google DeepMind, writes: "Today, we're excited to share the first real-world vulnerability discovered by the Big Sleep agent: an exploitable stack buffer underflow in SQLite, a widely used open source database engine." And: "We discovered the vulnerability and reported it to the developers in early October, who fixed it on the same day. Fortunately, we found this issue before it appeared in an official release, so SQLite users were not impacted." The successful run was "based on Gemini 1.5 Pro". They do not claim it is the first bug an AI ever found; their claim is narrower: "We believe this is the first public example of an AI agent finding a previously unknown exploitable memory-safety issue in widely used real-world software." They also write that earlier that year, at DARPA's AI Cyber Challenge, "Team Atlanta discovered a null-pointer dereference in SQLite". Their argument for defense: "Finding vulnerabilities in software before it's even released, means that there's no scope for attackers to compete". And their caveat: "we want to reiterate that these are highly experimental results. The position of the Big Sleep team is that at present, it's likely that a target-specific fuzzer would be at least as effective (at finding vulnerabilities)." A fuzzer is a program that feeds software large numbers of random or altered inputs to make it fail.

    DARPA's AI Cyber Challenge, AIxCC, ran for two years. Team Atlanta, of Georgia Tech, Samsung Research, KAIST and POSTECH, won the final. According to DARPA, the teams "also discovered 18 real, non-synthetic vulnerabilities that are being responsibly disclosed to open source project maintainers", "Teams submitted patches in an average of 45 minutes" and "Teams spent about $152 per competition task". The systems are to be shared: "All seven finalist teams’ CRSs will be made available as open-source software under a license approved by the Open Source Initiative." A CRS, a cyber reasoning system, finds and patches vulnerabilities without a person at the controls. DARPA also corrected itself: "A former version of this update stated the Final Competition included 70 synthetic vulnerabilities. Upon further review, the competition administrator determined the Final Competition included 63 synthetic vulnerabilities." DARPA's director says that the technology "will give defenders a much-needed edge in identifying and patching vulnerabilities at speed and scale."

    ### Who gains more?

    The International AI Safety Report 2026 gives no single answer. Under the heading "The offence-defence balance is critical but dynamic", it writes that "AI capabilities that allow an attacker to rapidly discover vulnerabilities can also be used by a defender to find and patch them first." And: "How this balance between offensive and defensive uses of AI evolves depends in part on choices about model access, research funding, and deployment standards." It names an asymmetry: "the lack of standard quality-assurance methods for AI tools makes it difficult for defenders to adopt them in critical sectors where reliability is essential, while attackers face no such constraints." And a dilemma for whoever sets the rules: "Since it can be difficult to distinguish helpful uses from harmful ones, overly aggressive safeguards such as preventing AI systems from responding to cyber-related requests can hamper defenders." So far, according to the report, "AI capabilities have primarily accelerated or scaled existing attack methods rather than created new kinds of attacks", and "general-purpose AI systems have not been reported to conduct end-to-end cyberattacks in the real world."

    Hendrycks and colleagues warn that the cost-benefit analysis of cybersecurity "may become lopsided if attackers eventually gain a larger menu of options that require negligible effort. In this new regime, attackers may gain the upper hand, like how attackers of ML systems currently have a large advantage over defenders." Their advice: "Since there may be less of a duality between offensive and defensive security in the future, we suggest that research focus on techniques that are clearly defensive." And their hedge: "The severity of this risk is speculative".

    Toby Shevlane and Allan Dafoe, of the Centre for the Governance of AI at the same institute, look at publishing AI research, in a paper of 2020: "While disclosure of software vulnerabilities often favours defence, this cannot be assumed for AI research." A fix for software can be spread to many systems at once; misuse of AI, they argue, more often calls for changes to how people deal with each other, which is often much more costly. In their words: "the vulnerabilities will be on average harder to patch than software vulnerabilities". Their conclusion, which they call tentative: "Overall, we tentatively argue that within AI, the security value of publication will be net negative in a significant fraction of cases, and more so than in computer security." The Big Sleep team, by contrast, hopes for "an asymmetric advantage for defenders", and aims to keep sharing its research, "keeping the gap between the public state-of-the-art and private state-of-the-art as small as possible."
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **systemic safety** | using ML to make the context in which AI systems work safer, such as cybersecurity and informed decision making; after Hendrycks and colleagues |
        | **offense-defense balance** | how easy it is to carry out an attack compared with defending against it, for one objective, form of conflict and set of actors; after Garfinkel and Dafoe |
        | vulnerability | a flaw in software that an attacker can use |
        | patch | a change to software that removes a vulnerability |
        | zero day | a vulnerability the defender does not know yet; `zero_days` in Exercise 1 |
        | OD-scaling | offensive-then-defensive scaling: more investment on both sides first favors the attacker, then the defender |
        | defensive accelerationism | focusing on technologies that defend against societal risks |
        | $\min(n, \text{attack})$ | the smaller of the two; in Python, `min(n, attack)` |
        """),
        L.watch_out("Garfinkel and Dafoe's model is idealized, and the numbers in this lesson are made up: they show what follows from the assumptions, not what will happen. Whether AI helps attackers or defenders more is, according to the International AI Safety Report 2026, still uncertain."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To judge whether an AI capability helps attack or defense more:

    1. Name the objective, the form of conflict and who is involved: the balance depends on all three.
    2. Ask what the same capability does for each side.
    3. Ask who gets to use it first, and whether one side must find everything while the other needs one gap.
    4. Ask whether more of it helps the attacker now and the defender later, as in OD-scaling.
    5. Look for what holds one side back: missing quality checks, costly repairs, limited access.
    6. Prefer uses that clearly help defense, and keep safeguards from shutting out defenders.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell works out how many vulnerabilities an attacker finds at three levels of effort, in software with 10 vulnerabilities. One unit of effort gives one expected discovery, as in Garfinkel and Dafoe's model. Python's `min` gives the smallest of the values it gets. The numbers are made up.
    """)
    return


@app.cell
def example_found():
    n = 10   # made-up: the number of vulnerabilities in the software
    for attack in [4, 10, 14]:
        found = min(n, attack)   # the smaller of the two: no side finds more vulnerabilities than there are
        print("effort", attack, "finds", found, "vulnerabilities")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With an effort of 4, the attacker finds 4 vulnerabilities; from an effort of 10 on, it finds all 10, and more effort finds no more. In Exercise 1, you add the defender: of the vulnerabilities the attacker finds, only the ones the defender has not found are zero days.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_zero_days():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. zero_days: what the attacker gains

        Garfinkel and Dafoe's idealized model in a function. Write `zero_days(n, attack, defend)`, which returns the expected number of zero days for `n` vulnerabilities, an attacker's effort `attack` and a defender's effort `defend`. Set `found` to the number of vulnerabilities the attacker finds, as in the worked example, and `unknown` to the share of all `n` vulnerabilities that the defender has not found; the defender, too, finds at most `n`. Return `found` times `unknown`. Replace the `...`. The check calls `zero_days(10, 4, 4)`, then other cases.
        """),
        L.task("The expected number of zero days for an attacker and a defender."),
    ])
    return


@app.cell(hide_code=True)
def exercise_zero_days():
    box_zero_days = L.code_box("zero_days", lines=4)
    box_zero_days
    return (box_zero_days,)


@app.cell(hide_code=True)
def check_zero_days(box_zero_days):
    result_zero_days = L.check("zero_days", box_zero_days.value)
    result_zero_days
    return (result_zero_days,)


@app.cell(hide_code=True)
def hints_zero_days():
    L.hints(
        "zero_days",
        [
            "Start small: 10 vulnerabilities, and each side finds 4. How many does the attacker find, and what share of all 10 has the defender not found?",
            "Set `found` as in the worked example. For `unknown`, take what the defender finds, never more than `n`, subtract it from `n`, and divide by `n`. Return `found` times `unknown`.",
            "`    unknown = (n - ___) / n`",
        ],
        why="`found = min(n, attack)`, `unknown = (n - min(n, defend)) / n` and `return found * unknown`. With 10 vulnerabilities and an effort of 4 on each side, the attacker finds 4, and the defender has not found 6 of the 10: 4 × 0.6 = 2.4 zero days.",
    )
    return


@app.cell(hide_code=True)
def text_after_zero_days():
    mo.md(r"""
    The more the defender finds, the fewer zero days: against an attacker's effort of 4, there are 2.4 at a defender's effort of 4, and none from 10 on, where the defender finds all. An attacker who does more than there is to find gains no more: at an effort of 14 against 3, the attacker finds all 10, and 7 of them are unknown to the defender. In Garfinkel and Dafoe's words, "any vulnerability the attacker and defender both discover does not become a zero day".
    """)
    return


@app.cell(hide_code=True)
def text_exercise_scaled():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. scaled: better tools for both

        Garfinkel and Dafoe treat better tools for finding vulnerabilities as more effort for both sides. Write `scaled(n, attack, defend, factors)`, which returns a list `result` with, for each `factor` in the list `factors`, the expected number of zero days when both `attack` and `defend` are multiplied by `factor`. The course gives you its own `zero_days`, so this works even if Exercise 1 is not done yet. In a `for` loop with `factor`, append what `zero_days` gives. Replace the `...`. The check calls `scaled(10, 2, 1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])`, an attacker who makes twice the defender's effort, then other cases.
        """),
        L.task("The expected number of zero days as better tools multiply the efforts of both sides."),
    ])
    return


@app.cell(hide_code=True)
def exercise_scaled():
    box_scaled = L.code_box("scaled", lines=5)
    box_scaled
    return (box_scaled,)


@app.cell(hide_code=True)
def check_scaled(box_scaled):
    result_scaled = L.check("scaled", box_scaled.value)
    result_scaled
    return (result_scaled,)


@app.cell(hide_code=True)
def hints_scaled():
    L.hints(
        "scaled",
        [
            "Start small: work out `zero_days(10, 2, 1)` and `zero_days(10, 4, 2)` by hand. What changed between the two calls, and how does the number of zero days change?",
            "Begin `result` as an empty list. In a `for` loop with `factor` over `factors`, call `zero_days` with `n` and the two efforts multiplied by `factor`, and append what it gives. After the loop, return `result`.",
            "`        result.append(zero_days(n, ___, defend * factor))`",
        ],
        why="`result = []`, then `for factor in factors:` with `result.append(zero_days(n, attack * factor, defend * factor))`; after the loop, `return result`. `zero_days(10, 2, 1)` is 2 × 0.9 = 1.8, and `zero_days(10, 4, 2)` is 4 × 0.8 = 3.2: both efforts doubled, and the attacker gains.",
    )
    return


@app.cell(hide_code=True)
def text_after_scaled():
    mo.md(r"""
    Better tools first help the attacker: the expected zero days rise from 1.8 at a factor of 1 to 5 at a factor of 5. Then they help the defender: from there on they fall, to 0 at a factor of 10, where the defender finds all 10 vulnerabilities, even though the attacker makes twice the effort all along. That is OD-scaling in this model. Whether real tools work this way, the model does not say: Garfinkel and Dafoe call it a plausible effect, "all else being equal". And they add that all else might not be equal: the vulnerabilities that highly optimized tools discover might be more correlated, so that both sides more often find the same ones, and by their analysis that would favor the defender. Their conclusion: "In all, the short-term impact of increased automation is ambiguous."
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: how many vulnerabilities

    Both sides make the same effort, and you change the number of vulnerabilities in the software. See up to which effort more effort still helps the attacker.
    """)
    return


@app.cell(hide_code=True)
def demo_vulns():
    vulns = L.explore("vulns", "n: the number of vulnerabilities",
                      lambda _v: C.demo_frame(_v, L.lang), start=10, values=C.VULNS, role="",
                      task="Find the smallest number of vulnerabilities at which the attacker still gains from more effort up to an effort of 10.")
    vulns
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With the same effort on both sides, the zero days are highest at an effort of half the number of vulnerabilities, and there they are a quarter of it: 1.25 at an effort of 2.5 for 5 vulnerabilities, 2.5 at 5 for 10, 5 at 10 for 20, and 10 at 20 for 40. So from 20 vulnerabilities on, more effort helps the attacker up to an effort of 10. The more vulnerabilities there are, the longer more effort keeps helping the attacker before it starts to help the defender.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_percent():
    mo.md(r"""
    ### Exercise 3. A number in DARPA's update

    DARPA writes about the final that the teams' systems "identified 86% of the competition’s synthetic vulnerabilities, an increase from 37% at semifinals, and patched 68% of the vulnerabilities identified". The counts are 63 vulnerabilities put in, 54 found and 43 patched.
    """)
    return


@app.cell(hide_code=True)
def choice_percent_cell():
    choice_percent = L.choice(
        "percent",
        question="What is the 68% a share of, by the counts?",
        options={
            "found": "43 of the 54 found, exactly as DARPA's sentence says",
            "right": "43 of the 63 put in; of the 54 found, about 80%",
            "old": "43 of the 70 that DARPA first counted",
        },
        correct="right",
        why={
            "right": "43 / 63 ≈ 0.68, and 43 / 54 ≈ 0.80. So the 68% is a share of all 63 vulnerabilities put in, not of the 54 found; DARPA's editor's note says that competitors \"discovered 86% of the synthetic vulnerabilities and patched 68%\". Of what they found, the systems patched about 80%. A percentage says little until you know what it is a share of.",
            "found": "43 / 54 is about 0.80, not 0.68. The 68% is 43 of all 63 vulnerabilities: 43 / 63 ≈ 0.68. The sentence says \"of the vulnerabilities identified\", but the counts do not fit that reading; DARPA's editor's note says \"patched 68%\" of the synthetic vulnerabilities.",
            "old": "43 / 70 is about 0.61. DARPA corrected the 70 to 63, and 43 / 63 ≈ 0.68: the 68% is a share of all 63 vulnerabilities put in. Of the 54 found, the systems patched about 80%.",
        },
    )
    choice_percent
    return (choice_percent,)


@app.cell(hide_code=True)
def text_exercise_voice():
    mo.md(r"""
    ### Exercise 4. A voice on the phone

    A lab publishes a method that copies anyone's voice from a few seconds of recorded speech. In computer security, publishing a vulnerability often helps defense: once it is known, a fix can be spread to every system.
    """)
    return


@app.cell(hide_code=True)
def choice_voice_cell():
    choice_voice = L.choice(
        "voice",
        question="Does publishing help defense here too, by Shevlane and Dafoe's reasoning?",
        options={
            "yes": "Yes: once the method is known, everyone can prepare, as with software",
            "right": "Not necessarily: what it exploits, trust in a familiar voice, is hard to patch",
            "always": "No: by their argument, publishing AI research always helps attackers more than defenders",
        },
        correct="right",
        why={
            "right": "The method exploits something people do every day: they take the sound of a voice as evidence of who is speaking. Unlike a line of code, that habit cannot be fixed with an update. Shevlane and Dafoe give a similar example: criminals who used generated speech to fake the voice of a company's CEO over the phone. By their reasoning, publishing then has less defensive value than in software.",
            "yes": "Knowing about the method helps some, but what it exploits is not code: people take a familiar voice as evidence of who is speaking, and no update changes that habit. Shevlane and Dafoe argue that such vulnerabilities are \"on average harder to patch than software vulnerabilities\". See *Who gains more?* in *The idea*.",
            "always": "Their claim is narrower and, in their own word, tentative: the security value of publication will be net negative \"in a significant fraction of cases\", not always. Here the method exploits trust in a familiar voice, which no update fixes, so publishing helps defense less than in software.",
        },
    )
    choice_voice
    return (choice_voice,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Races and selection](/course/open/rai-race-and-selection?lang=en) showed how competition can push toward less care. This lesson looked at the other side: AI used to make the systems around it safer, which Hendrycks and colleagues call systemic safety. The same capability that finds a vulnerability for an attacker can find it for a defender, and in Garfinkel and Dafoe's model, better tools first help the attacker and then the defender. Whether AI helps attackers or defenders more is uncertain, and according to the International AI Safety Report 2026 it depends in part on choices. The next lesson, *The arguments for catastrophic risk*, looks at the arguments that AI could lead to a catastrophe.

    **An everyday example.** A lock-testing kit shows which locks on a street are weak. A burglar who gets it first finds a way in. A resident who uses it first replaces the weak locks, and the burglar finds nothing. Who gains from the kit depends on who uses it first, and on how many weak locks there are.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_percent, choice_voice, result_scaled, result_zero_days):
    L.wrap_up(result_zero_days, result_scaled, choice_percent, choice_voice)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
