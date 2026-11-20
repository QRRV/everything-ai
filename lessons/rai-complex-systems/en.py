# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-complex-systems. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-complex-systems", "en")
    C = L.checks   # a made-up market with a sell algorithm, and eight made-up models that add numbers


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Drifting into failure, and black swans](/course/open/rai-drift-and-tails?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_drift():
    quick_drift = L.choice(
        "quick-drift",
        question="A team watches a model's error rate, with an alarm at 2%. Each time the rate goes a little over the line and nothing bad happens, the team moves the line up a little. After a year, it is at 5%. What is this?",
        options={
            "swan": "A black swan: nobody saw it coming",
            "right": "Drift into failure",
            "tail": "A long tail: a rare value far from the rest, which weighs more than all the others",
        },
        correct="right",
        why={
            "right": "Each step is accepted because the last one did no harm, and so the line creeps up: that is drift into failure. Nothing has gone wrong yet, but the room the alarm was meant to give is gone.",
            "swan": "A black swan is a rare event with a large impact that most people did not see coming. Here nothing has happened yet: the line moves up step by step, each step accepted because the last one did no harm. That is drift into failure. See [Drifting into failure, and black swans](/course/open/rai-drift-and-tails?lang=en).",
            "tail": "A long tail is about how big rare values can get. Here the line moves up step by step, each step accepted because the last one did no harm: that is drift into failure. See [Drifting into failure, and black swans](/course/open/rai-drift-and-tails?lang=en).",
        },
    )
    quick_drift
    return (quick_drift,)


@app.cell(hide_code=True)
def quick_check_tail():
    quick_tail = L.choice(
        "quick-tail",
        question="The damage of incidents in a system has a long tail. In ten years, the worst one cost 2 million. What does that say about the worst incident of the next ten years?",
        options={
            "right": "Little: a single new one can be far bigger",
            "same": "It will cost about 2 million too",
            "cap": "At most 2 million, since that is the worst in ten years of data",
        },
        correct="right",
        why={
            "right": "Under a long tail, one new value can be bigger than all the earlier ones together, so the worst so far is no ceiling. Ten years of data say little about the next incident.",
            "same": "That would hold for a thin tail, such as height. Under a long tail, one new value can be far bigger than all before it, so the worst so far says little about the next. See [Drifting into failure, and black swans](/course/open/rai-drift-and-tails?lang=en).",
            "cap": "The worst so far is no ceiling: under a long tail, one new incident can be bigger than all the earlier ones together. See [Drifting into failure, and black swans](/course/open/rai-drift-and-tails?lang=en).",
        },
    )
    quick_tail
    return (quick_tail,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The lessons so far looked at one system at a time: a model, its safeguards, its tests. But an AI system does not run alone. It answers people who change what they ask, it trades in markets where other programs trade too, and what it makes ends up in the world it learns from. When parts react to each other like that, the whole can do things that no part does on its own, and a fix in one place can make things worse in another. This lesson shows what that looks like, in a market crash and in the way abilities of language models are measured, and why safety is then not a bug you fix once.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="On 6 May 2010, US stock prices crashed and recovered within minutes. In 15 seconds of the crash, high-frequency traders, firms whose computers trade very fast, traded over 27,000 futures contracts. How many more did they buy than they sold?",
        options={
            "right": "About 200",
            "half": "About 13,500: half of what they traded",
            "all": "About 27,000: they only bought",
        },
        reveal="**About 200.** The joint report of the staffs of the CFTC and the SEC, two US market regulators: \"Between 2:45:13 and 2:45:27, HFTs traded over 27,000 contracts, which accounted for about 49 percent of the total trading volume, while buying only about 200 additional contracts net.\" They bought and sold the same contracts back and forth, and the report calls it a \"hot-potato\" volume effect. A lot was traded, but almost nobody kept what they bought.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Many parts that react to each other

    Dan Hendrycks's textbook *Introduction to AI Safety, Ethics, and Society* (2024) begins its chapter on complex systems with: "Artificial intelligence systems and the societies they operate within belong to the class of complex systems." Its working definition: "Complex systems are systems of many interconnected components that collectively exhibit emergent features, which cannot, in practice, be derived from a reductive analysis of the system in terms of its isolated components."

    The key word is **emergence**: "the appearance of striking, system-wide features that cannot be found in any of the system’s components." Ants show it: "A group of 100 army ants placed on the ground behaves not like an enfeebled colony but rather like no colony at all". The ants walk in circles until they die. With tens of thousands of them, the colony starts to behave like one intelligent whole. No single ant has that behavior; it comes from how they react to each other.

    Besides emergence, the textbook names "self-organization, feedback and nonlinearity, criticality, adaptive behavior, distributed functionality, and scalable structure." Feedback "refers to circular processes in which a system and its environment affect one another": what comes out goes back in, a feedback loop. And the textbook adds a caveat itself: "Complexity can be a feature of our understanding of the system rather than the system itself." A gas was hard to explain from its molecules, until decades of physics worked it out.

    ### The flash crash of 6 May 2010

    On that afternoon, major US stock indices, "each already down over 4% from their prior-day close, suddenly plummeted a further 5-6% in a matter of minutes before rebounding almost as quickly." And: "Over 20,000 trades across more than 300 securities were executed at prices more than 60% away from their values just moments before." So writes the joint report of the staffs of the CFTC and the SEC, of 30 September 2010.

    At 2:32 p.m., a large trader, a mutual fund complex, began to sell 75,000 E-Mini contracts, futures that follow the S&P 500 stock index, "valued at approximately $4.1 billion". It used a sell algorithm set to "target an execution rate set to 9% of the trading volume calculated over the previous minute, but without regard to price or time." An earlier sale of that size had taken "more than 5 hours"; this one took 20 minutes.

    High-frequency traders bought at first, and then passed contracts back and forth among themselves: the hot potato from the guess. All that trading counted as volume, and "The Sell Algorithm used by the large trader responded to the increased volume by increasing the rate at which it was feeding the orders into the market". More selling, more trading, more volume, more selling: a feedback loop. At 2:45:28 p.m., the exchange paused trading in the E-Mini for five seconds, "in order to prevent a cascade of further price declines." When trading resumed, "prices stabilized and shortly thereafter, the E-Mini began to recover".

    No part of this was built to crash a market. Among the key lessons in the report: "under stressed market conditions, the automated execution of a large sell order can trigger extreme price movements, especially if the automated execution algorithm does not take prices into account." Moreover, it writes, "the interaction between automated execution programs and algorithmic trading strategies can quickly erode liquidity and result in disorderly markets." After the crash came a pilot program, at first for the stocks in the S&P 500 and until 10 December 2010: it paused trading in a single stock "for five minutes if that security has experienced a 10% price change over the preceding five minutes". That is a brake that works whatever the cause. The report speaks of algorithms, not of AI. The textbook uses the flash crash as an example of how quickly an error in an automated system can escalate, and speaks of "a feedback loop between automated trading algorithms".

    ### Abilities that seem to appear

    For language models, the word has a second meaning, and whether such abilities really appear is disputed. Jason Wei and colleagues (2022): "An ability is emergent if it is not present in smaller models but is present in larger models." On a chart of performance against size, "performance is near-random until a certain critical threshold of scale is reached". They also note that the scale at which an ability first shows up "is not an immutable property of the ability." The textbook states it as a fact: "Larger LLMs, however, spontaneously gain qualitatively new capabilities, such as translating text or performing three-digit arithmetic."

    Rylan Schaeffer and colleagues (2023) doubt it: "emergent abilities may be creations of the researcher’s choices, not a fundamental property of the model family on the specific task." Their example is a metric that scores an addition 1 "if all L output digits exactly match all target digits", and 0 otherwise: an exact match. If each digit is right or wrong independently of the others, the chance of a fully right answer falls fast with its length. By their count, "> 92% of emergent abilities on BIG-Bench tasks" appear under one of two metrics: exact match and a multiple-choice score, which both give an answer 1 or 0. They set a limit to their own claim: "nothing in this paper should be interpreted as claiming that large language models cannot display emergent abilities".

    ### Fixes that bite back

    In a paper published in 1973, Horst Rittel and Melvin Webber, a professor of design and a professor of city planning at Berkeley, wrote about the problems of planning and social policy: "They are ‘wicked’ problems, whereas science has developed to deal with ‘tame’ problems." Wicked does not mean evil: they use the word "in a meaning akin to that of ‘malignant’ (in contrast to ‘benign’) or ‘vicious’ (like a circle) or ‘tricky’ (like a leprechaun)". They list ten properties, among them "1. There is no definitive formulation of a wicked problem", "2. Wicked problems have no stopping rule" and "3. Solutions to wicked problems are not true-or-false, but good-or-bad". Such problems are not solved once: "At best they are only re-solved--over and over again."

    Rittel and Webber wrote about planning, not about AI. The textbook uses the term for AI: "Wicked problems usually arise in complex systems and often involve a social element." And it gives complexity as the reason: "Owing to the complex nature of the systems involved, AI safety is a wicked problem." For the same reason, it writes, "making AI safe is not like solving a mathematical problem or fixing a watch." A fix can also cause new problems: "some proposed solutions may have negative side effects and create other issues, even if they reduce the targeted wicked problem." And some attempts "will be ineffective, have negative side effects, or even backfire."

    On one point they disagree. Rittel and Webber: "5. Every solution to a wicked problem is a ‘one-shot operation’; because there is no opportunity to learn by trial-and-error, every attempt counts significantly". The textbook warns that "There is often a risk involved in attempting to solve a wicked problem", but also writes: "Learning how to make AIs safe will require some trial and error."
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **emergence** | behavior of a whole that none of its parts has on its own, such as that of an ant colony |
        | **wicked problem** | a problem without a final formulation or a final solution, where every attempt changes the problem; after Rittel and Webber (1973) |
        | complex system | a system of many parts that react to each other, whose behavior you cannot work out from the parts alone |
        | feedback loop | what comes out goes back in, as the volume came back to the sell algorithm |
        | high-frequency trader | a trading firm whose computers trade very many contracts, but keep few of them for long |
        | emergent ability | an ability that larger language models seem to have and smaller ones do not; whether the jump is real is disputed |
        | exact match | a metric that counts an answer as right only when all of it is right |
        """),
        L.watch_out("Emergence has two meanings here. In a complex system, it is behavior of the whole that no part has. For language models, it is an ability that larger models have and smaller ones do not. Whether such abilities really appear in language models is disputed."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To look at an AI system as part of a complex system:

    1. Write down who and what react to the system: its users, other programs, the people it affects, and the data it learns from.
    2. Look for loops: what comes out and goes back in, as the volume came back to the sell algorithm.
    3. Test parts together, not only one by one: behavior can emerge from how they react to each other.
    4. Build in brakes that work whatever the cause, such as a pause when things move too fast.
    5. Expect a fix to change the system, and watch what happens after it, not only before.
    6. Count on working on it again: a wicked problem is not solved once.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The sell algorithm of the flash crash sold 9% of the volume of the minute before. The cell works out what such an algorithm sells in a market whose volume rises over four minutes. The volumes are made up.
    """)
    return


@app.cell
def example_rule():
    rate = 0.09                         # the algorithm sells this share of the previous minute's volume
    traded = [4000, 6000, 9000, 14000]  # contracts traded in four minutes, made up
    for minute in range(1, len(traded)):
        sold = rate * traded[minute - 1]
        print("in minute", minute + 1, "the algorithm sells", round(sold), "contracts")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The more is traded, the more the algorithm sells: 360, 540, and then 810 contracts. It does not look at the price, only at the volume. But who makes the volume? Partly the algorithm itself, once what it sells is traded on. In Exercise 1, you close that loop.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_volumes():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. volumes: a loop that feeds itself

        A made-up, simplified market with the rule from the worked example. Each minute, the other traders trade `base` contracts. The sell algorithm sells `rate` times the volume of the minute before. The fast traders pass each contract it sells back and forth, so in that same minute each one is traded `echo` times. Write `volumes(base, rate, echo, minutes)`, which returns a list `result` with the volume of each minute. Begin `volume` at `base`. In a `for` loop with `minute` over `range(minutes)`, append `volume` to `result`, work out `sold`, what the algorithm sells in the next minute, and then that next minute's `volume`. Replace the `...`. The check calls `volumes(1000, 0.09, 12, 8)`, then other cases.
        """),
        L.task("The volume of each minute in the made-up market."),
    ])
    return


@app.cell(hide_code=True)
def exercise_volumes():
    box_volumes = L.code_box("volumes", lines=8)
    box_volumes
    return (box_volumes,)


@app.cell(hide_code=True)
def check_volumes(box_volumes):
    result_volumes = L.check("volumes", box_volumes.value)
    result_volumes
    return (result_volumes,)


@app.cell(hide_code=True)
def hints_volumes():
    L.hints(
        "volumes",
        [
            "Put the volume in the list before you change it: the first minute has only the other traders' `base`.",
            "Begin `volume` at `base` and `result` as an empty list. In a `for` loop over `range(minutes)`, append `volume`, set `sold` to `rate` times `volume`, and then set `volume` to the next minute's volume: the other traders' `base`, plus each sold contract traded `echo` times. After the loop, return `result`.",
            "`        volume = base + ___`",
        ],
        why="`volume = base` and `result = []`, then `for minute in range(minutes):` with `result.append(volume)`, `sold = rate * volume` and `volume = base + echo * sold`; after the loop, `return result`. In the first case, minute 1 has 1,000 contracts; the algorithm sells 90 of them, each traded 12 times, so minute 2 has 1,000 + 1,080 = 2,080, and so on.",
    )
    return


@app.cell(hide_code=True)
def text_after_volumes():
    mo.md(r"""
    With an echo of 12, every minute adds more than the minute before: from 1,000 to 10,637 contracts in eight minutes. With an echo of 5, the same market settles near 1,818. The algorithm's rule is the same in both; only how strongly the others react to it differs. In the flash crash, part of the volume that the algorithm followed was the high-frequency traders' hot potato.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_accuracies():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. accuracies: one digit or all of them

        Schaeffer and colleagues' example: a model adds numbers, and an answer of `length` digits counts as right only when every digit is right, an exact match. The chances that eight made-up models, each bigger than the one before, get a single digit right are in a NumPy array `per_digit`. Assume, as they do, that each digit is right or wrong independently of the others. Write `accuracies(per_digit, lengths)`, which returns a list `rows`: for each `length` in the list `lengths`, an array with every model's chance of an exact match on an answer of that length. Use a `for` loop with `length`, and let NumPy work on all models at once. Replace the `...`. The check calls `accuracies(np.array([0.5, 0.6, 0.7, 0.8, 0.88, 0.94, 0.97, 0.99]), [1, 10])`, then other cases.
        """),
        L.task("Every model's chance of an exact match, for each length."),
    ])
    return


@app.cell(hide_code=True)
def exercise_accuracies():
    box_accuracies = L.code_box("accuracies", lines=5)
    box_accuracies
    return (box_accuracies,)


@app.cell(hide_code=True)
def check_accuracies(box_accuracies):
    result_accuracies = L.check("accuracies", box_accuracies.value)
    result_accuracies
    return (result_accuracies,)


@app.cell(hide_code=True)
def hints_accuracies():
    L.hints(
        "accuracies",
        [
            "Start small: if one digit is right with a chance of 0.9, and each digit is right or wrong independently, what is the chance that two digits are both right?",
            "Begin `rows` as an empty list. In a `for` loop over `lengths`, append the chance that all `length` digits are right: `per_digit` multiplied by itself `length` times. NumPy does that for all models at once, with one power of the array. After the loop, return `rows`.",
            "`        rows.append(___)`",
        ],
        why="`rows = []`, then `for length in lengths:` with `rows.append(per_digit ** length)`; after the loop, `return rows`. Two digits right at 0.9 each: 0.9 × 0.9 = 0.81; ten digits: 0.9 to the power 10, about 0.35. In the first case, the row for 10 digits is 0.001, 0.006, 0.028, 0.107, 0.279, 0.539, 0.737 and 0.904.",
    )
    return


@app.cell(hide_code=True)
def text_after_accuracies():
    mo.md(r"""
    One digit goes up steadily from model to model, from 0.5 to 0.99. All 10 digits right stays near 0 for the first three models, at 0.001, 0.006 and 0.028, and then rises to 0.904. On a chart of exact match, that looks like an ability that appears at a certain size; per digit, the models get better step by step. That is Schaeffer and colleagues' point: the same models can look like a jump or like steady progress, depending on the metric. It does not show that no ability can ever appear suddenly, as they say themselves.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: how strongly the market reacts

    The made-up market from Exercise 1, over twenty minutes. Turn up the echo, how often each contract the algorithm sells is traded on, and see when the volume stops settling.
    """)
    return


@app.cell(hide_code=True)
def demo_echo():
    echo = L.explore("echo", "echo: how often each sold contract is traded on",
                     lambda _v: C.demo_frame(_v, L.lang), start=5, values=C.ECHOES, role="",
                     task="Find the smallest echo at which each minute adds more than the minute before.")
    echo
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    From an echo of 12, 0.09 × 12 = 1.08 is above 1: each minute adds more than the one before, and the volume runs away. At 11, 0.99 is just below 1: in the long run the volume would settle, though only at about 100,000, and in these twenty minutes it is still climbing. The algorithm's rule and the other traders' 1,000 contracts stayed the same; only the echo, how often the fast traders pass on what it sells, went up by one step. A small change in one part gave a very different whole.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_agents():
    mo.md(r"""
    ### Exercise 3. A thousand agents

    A company tests its trading agent, an AI system that buys and sells on its own, alone in a simulated market, and it behaves well. Then a thousand copies of it go live, in a market where agents of other companies trade too.
    """)
    return


@app.cell(hide_code=True)
def choice_agents_cell():
    choice_agents = L.choice(
        "agents",
        question="What do the tests of one agent alone tell you about that market?",
        options={
            "enough": "Enough, if each agent behaves well",
            "right": "Not much: together, the agents can do what no single agent does",
            "more": "Everything, as long as the simulated market was big enough and ran long enough",
        },
        correct="right",
        why={
            "right": "Tests of one agent show what it does alone. A market of agents is a complex system: as in the flash crash, where each part followed its own rule, the trouble can come from how the rules feed each other. The ants show it the other way round: 100 of them behave like no colony at all, tens of thousands like one whole. Test the agents together, and build in brakes that work whatever the cause.",
            "enough": "That each part behaves well does not make the whole behave well. In the flash crash, each algorithm did what it was set to do; the report's lessons point both to a large order sold without regard to price and to how the automated programs interacted. See *The flash crash of 6 May 2010* in *The idea*.",
            "more": "A bigger simulation with one agent still has one agent in it. What a thousand agents do together, and with the agents of other companies, can emerge only when they meet. See *Many parts that react to each other* in *The idea*.",
        },
    )
    choice_agents
    return (choice_agents,)


@app.cell(hide_code=True)
def text_exercise_backfire():
    mo.md(r"""
    ### Exercise 4. Driving on fewer days

    In 1989, Mexico City tackled its traffic and air pollution with a program called Hoy No Circula: depending on the last number on their license plate, people could drive only on certain days of the week. At first, emissions dropped.
    """)
    return


@app.cell(hide_code=True)
def choice_backfire_cell():
    choice_backfire = L.choice(
        "backfire",
        question="What happened next, according to the textbook?",
        options={
            "stayed": "Emissions stayed low, since fewer cars drove each day",
            "right": "Emissions rose again, even above the level before the rule",
            "buses": "Emissions kept falling for years, as people moved to buses and trains for good",
        },
        correct="right",
        why={
            "right": "The textbook: emissions \"soon rose again, actually surpassing the pre-intervention levels.\" According to a study it cites, people bought extra cars so they could drive on more days, and \"the extra cars people bought tended to be cheaper, older, more polluting ones\". The city changed one rule, the people in it reacted, and the problem came back bigger: a fix that backfired because the system reacted to it.",
            "stayed": "Fewer cars drove each day at first, but people reacted to the rule: a study found that they bought extra cars so they could drive on more days. Emissions rose again, above the level before the rule. See *Fixes that bite back* in *The idea*.",
            "buses": "The textbook reports the opposite: people bought extra cars, often older ones that polluted more, and emissions rose above the level before the rule. See *Fixes that bite back* in *The idea*.",
        },
    )
    choice_backfire
    return (choice_backfire,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Drifting into failure, and black swans](/course/open/rai-drift-and-tails?lang=en) showed how one system slides toward an accident, step by step. This lesson looked at many parts that react to each other. Behavior that no part has, such as a flash crash, can emerge from how they interact; a jump on a chart can come from the metric rather than from the model; and a fix can change the system it was meant to fix. The textbook calls AI safety a wicked problem because AI and the society around it are complex systems: not solved once, but worked on again and again. The next lesson, *Races and selection*, looks at one such interaction: companies and states that compete.

    **An everyday example.** A traffic jam can move backward along a road while every car in it drives forward. No driver makes the jam; it comes from how each one reacts to the car ahead.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_agents, choice_backfire, result_accuracies, result_volumes):
    L.wrap_up(result_volumes, result_accuracies, choice_agents, choice_backfire)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
