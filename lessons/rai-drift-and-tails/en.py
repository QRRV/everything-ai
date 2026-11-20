# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-drift-and-tails. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-drift-and-tails", "en")
    C = L.checks   # a limit that stretches, a mean after each value, and a thin tail beside a long one


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [Safeguards in a row](/course/open/rai-swiss-cheese?lang=en), and one about [Random numbers and seeds](/course/open/np-random?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_floor():
    quick_floor = L.choice(
        "quick-floor",
        question="Three safeguards each miss 10% of the ordinary harmful requests, independently, and a trick in 1% of the requests fools all three. What lowers the share that gets through the most?",
        options={
            "fourth": "A fourth safeguard that also misses 10%",
            "right": "Closing the trick's hole in one safeguard",
            "order": "Running the three safeguards in another order",
        },
        correct="right",
        why={
            "right": "The trick sets a floor: its 1% gets through however many safeguards stand in a row. Once one safeguard catches the trick too, the requests with the trick have to get past it like any other request.",
            "fourth": "A fourth safeguard lowers the ordinary share from 0.1% to 0.01%, but the 1% with the trick still gets through every safeguard. See [Safeguards in a row](/course/open/rai-swiss-cheese?lang=en).",
            "order": "With independent safeguards, the order changes nothing: the chances multiply the same way, and the trick still fools all three. See [Safeguards in a row](/course/open/rai-swiss-cheese?lang=en).",
        },
    )
    quick_floor
    return (quick_floor,)


@app.cell(hide_code=True)
def quick_check_seed():
    quick_seed = L.choice(
        "quick-seed",
        question="A cell runs `rng = np.random.default_rng(7)` and then `rng.random(size=3)`. You run the cell twice. What do you get?",
        options={
            "right": "The same three numbers both times",
            "new": "Three new numbers each time you run it",
            "whole": "Three whole numbers below 7",
        },
        correct="right",
        why={
            "right": "A generator with a fixed seed gives the same numbers on every run, so a result can be repeated. That is why this lesson's draws are the same for everyone.",
            "new": "With a fixed seed, the generator starts in the same state on every run, so it gives the same numbers. See [Random numbers and seeds](/course/open/np-random?lang=en).",
            "whole": "`rng.random` gives numbers between 0 and 1; whole numbers come from `rng.integers`. The 7 is the seed, not a limit. See [Random numbers and seeds](/course/open/np-random?lang=en).",
        },
    )
    quick_seed
    return (quick_seed,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A made-up company runs a chatbot that answers millions of questions a day. Its incident log is full of small mishaps: a wrong date here, a rude reply there. Then one answer spreads across the internet and does more harm than all the earlier incidents together.

    That is not just bad luck. Two ideas from safety engineering explain it: systems slide toward failure through small steps that each look fine, and in some domains one rare event outweighs everything before it. This lesson gives both, with the Space Shuttle Columbia as a documented case and NumPy to draw numbers with a long tail.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="A game: you win $15 with a chance of 99.7%, and you lose $10,000 with a chance of 0.3%. What do you win or lose per play, on average?",
        options={
            "most": "Win close to $15 per play",
            "five": "Win about $5 per play",
            "right": "Lose about $15 per play",
        },
        reveal="**Lose about $15 per play.** 0.997 × 15 + 0.003 × (−10,000) = 14.955 − 30 = −15.045. You win in 997 of 1,000 plays, and the three losses eat all of it and more. The textbook uses this game: with a 0.1% chance of losing, you win $4.985 per play on average; at 0.3% you lose $15.045. A change of 0.2 percentage points in one rare chance flips the sign.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Drifting, one small step at a time

    Dan Hendrycks's textbook *Introduction to AI Safety, Ethics, and Society* (2024) describes several models of how accidents happen. In its account of Jens Rasmussen's framework, accidents are "usually the culmination of many different factors, rather than a single root cause", and "behaviors and conditions can gradually ‘migrate’ over time, due to environmental pressures." Its account of Sidney Dekker's *Drift into Failure* adds: "Decrementalism is the deterioration of system processes through a series of small changes." Each step is small: "Many relatively minor decisions can combine to lead to a major difference in risk." And the textbook warns: "This is often only realized when an accident happens, at which point it is too late."

    Why does no one stop it? The textbook gives a reason in its account of the STAMP model: "If being less safety-conscious does not quickly lead to an accident, developers might start to think that safety-consciousness is unnecessary." That is **drift into failure**: systems "often ‘drift into failure’ through an accumulation and combination of many seemingly minor events, none of which would be catastrophic alone." For AI, the textbook says it carefully: "It is possible that many relatively minor events could accumulate, leading us to drift toward an existential risk."

    ### Columbia: a deviation that became normal

    On February 1, 2003, the Space Shuttle Columbia broke up during re-entry, and its crew of seven died. The Columbia Accident Investigation Board (August 2003) found the physical cause: a piece of insulating foam that came off the External Tank 81.7 seconds after launch and struck the left wing. Foam had come off before. The Board writes: "The history of foam-problem decisions shows how NASA first began and then continued flying with foam losses, so that flying with these deviations from design specifications was viewed as normal and acceptable." And: "Over the course of 113 missions, foam-shedding and other debris impacts came to be regarded more as a turnaround or maintenance issue, and less as a hazard to the vehicle and crew." It uses the sociologist Diane Vaughan's name for it: the normalization of deviance.

    The Board describes how the accepted range grew. Once the first deviations were accepted, later foam strikes "were not defined as signals of danger, but as evidence that the design was now acting as predicted", and the engineering experience base that took in ever worse anomalies "functioned as an elastic waistband, expanding to hold larger deviations from the original design." And: "Anomalies that did not lead to catastrophic failure were treated as a source of valid engineering data that justified further flights." It also quotes Richard Feynman on the Challenger accident of 1986, where the seals in the joints of the rocket boosters, the O-rings, had shown erosion on earlier flights. Feynman describes the same error: "The acceptance and success of these flights is taken as evidence of safety." Then Feynman turns it around: "Erosion was a clue that something was wrong."
    """)
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Tail events and black swans

    The textbook defines: "Tail events are events that happen rarely, but have a considerable impact when they do." Examples it gives are the financial crisis of 2007 and 2008, COVID-19, and the internet. And: "A tail event often changes the mean but not the median." The risk of such rare, extreme outcomes is called **tail risk**: when the tail is long, those few outcomes can make up most of the total.

    Whether a domain has tail events depends on its distribution. "In thin-tailed distributions, tail events do not exist. Examples of thin-tailed distributions include human characteristics such as height, weight, and intelligence." Long-tailed examples "include book sales, earthquake magnitude, and word frequency". In the textbook's table, in a long tail "A single event can escalate to become much bigger than many others put together." It warns that "not all distributions neatly fit into a dichotomy": the table gives simplified caricatures. M. E. J. Newman gives a sense of the difference in a paper on such distributions: the tallest and the shortest adult men in the Guinness Book of Records when Newman wrote, 272 cm and 57 cm, differ by a factor of 4.8; the largest city and the smallest town in the US differ in population by a factor of at least 150,000.

    "A black swan is a tail event that was largely unpredictable to most people before it happened", the textbook writes, and adds: "Note that not all tail events are black swans". A big earthquake that seismologists expect in California would be a tail event, but not a black swan. The name: "in 1697, a group of Dutch explorers encountered a black species of swan in Australia", though the metaphor itself is older. The textbook works with this loose definition, "a highly impactful event that is largely unexpected for most people", and not with a strict one, an event nobody at all could anticipate. Nassim Nicholas Taleb, whose book *The Black Swan* the textbook cites, stresses in a later book, *Statistical Consequences of Fat Tails*, that it depends on who is looking: "Black swans are observer-dependent: a Black Swan for the turkey is a White Swan for the butcher."

    ### Why the past says little about the next extreme

    With a long tail, an average from the past is a poor guide. The textbook: "a single extreme observation can change the average by several orders of magnitude". And from its examples of black swans it concludes that "we cannot necessarily make an accurate calculation of future risk based on long-running historical data." In the same later book, Taleb writes that "future extremes will not be similar to past extremes under fat tails, and such dissimilarity increases with fat tailedness". And the textbook: "An absence of evidence is not strong evidence of absence." For some long tails the average never settles: in Newman's fits for the intensity of wars and of solar flares, the mean is infinite.

    Newman gives a recipe for numbers with a long tail. Take a random number r between 0 and 1, and compute (1 − r) to the power −1/1.5: mostly a little above 1, now and then huge. With Newman's formula, the mean of this recipe in the long run is 3, but it gets there slowly.

    ### What helps

    Against drift, the textbook's lesson from Rasmussen is that "it is essential to continually enforce safety boundaries and avoid the system migrating into unsafe states". Against true black swans "we cannot put any specific defenses in place against them", but stress tests, horizon scanning, red teaming (attacking a system on purpose to find its weak spots) and what it calls safety imagination can turn some of them into risks that are on the radar, even if they are not understood. For AI, it argues, in hedged words, that "Complex systems may be more likely to entail a risk of black swans" and that "New systems may be more likely to present black swans."
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **drift into failure** | a system sliding toward an accident through many small steps, each accepted because the last one did no harm |
        | **tail risk** | the risk of rare, extreme outcomes far out in the tail of a distribution; with a long tail they can make up most of the total risk |
        | normalization of deviance | a deviation from the design that comes to count as normal because it has not caused an accident yet |
        | tail event | an event that happens rarely but has a considerable impact when it does |
        | black swan | a tail event that most people did not see coming; who is surprised depends on who is looking |
        | thin tail | a distribution without tail events, such as human height: no value is far from the rest |
        | long tail | a distribution in which one value can be bigger than many others together, such as book sales |
        | `x ** (-1 / 1.5)` | x to the power −1/1.5 in Python: one divided by x to the power 1/1.5 |
        """),
        L.watch_out("A long run without an accident is not evidence of safety. Under drift, the risk grows while nothing happens, and under a long tail, the worst so far says little about the next."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Write down the design limit before you start, and keep it apart from what has come to count as normal.
    2. For every deviation, ask whether it is accepted because it is understood, or only because nothing went wrong yet.
    3. Look at a distribution before you trust an average: compare the mean with the median, and the largest value with the total.
    4. When one value can be bigger than many others together, treat the largest so far as a lower bound for the worst still to come, not an upper bound.
    5. Against the rare events you cannot list, use stress tests, horizon scanning and red teaming, and keep the safety boundaries in force.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell draws 1,000 made-up heights, each 140 plus 80 times the mean of 12 random numbers, so they lie between 140 and 220 cm, around 180. It also draws 1,000 made-up incident sizes with Newman's recipe. For both, it prints the mean, the median and the largest value.
    """)
    return


@app.cell
def example_tails():
    import numpy as np

    rng = np.random.default_rng(7)
    heights = 140 + 80 * rng.random(size=(1000, 12)).mean(axis=1)   # cm: a thin tail
    sizes = (1 - rng.random(size=1000)) ** (-1 / 1.5)               # incidents: a long tail
    for name, values in [("heights", heights), ("incidents", sizes)]:
        print(name, round(values.mean(), 1), round(np.median(values), 1), round(values.max(), 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The heights have mean 180.1 and median 179.8, and the tallest is 199.9 cm, 11% above the mean. The incidents have mean 2.9 and median 1.6, and the largest is 177.4: about 60 times the mean, and on its own about 6% of the total of all 1,000. In the exercises you write how an accepted limit stretches, and the mean after each value, which shows what one large value does.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_stretch():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. stretch: the elastic waistband

        Each flight has a deviation from the design, such as how much foam came off. At first, the limit of what counts as acceptable is the design limit. When a flight has a bigger deviation and nothing goes wrong, that deviation becomes the new limit. Write `stretch(deviations, design)`. It gets the deviations of the flights in order and the design limit, and returns a list `accepted` with the accepted limit after each flight. Keep the limit in a variable `limit` and go through the flights with a `for` loop, with `deviation` for each one. Replace the `...`. The check calls `stretch([0.5, 1.4, 1.2, 2.0, 1.8, 3.1], 1)`, then other flights.
        """),
        L.task("The accepted limit after each flight."),
    ])
    return


@app.cell(hide_code=True)
def exercise_stretch():
    box_stretch = L.code_box("stretch", lines=8)
    box_stretch
    return (box_stretch,)


@app.cell(hide_code=True)
def check_stretch(box_stretch):
    result_stretch = L.check("stretch", box_stretch.value)
    result_stretch
    return (result_stretch,)


@app.cell(hide_code=True)
def hints_stretch():
    L.hints(
        "stretch",
        [
            "The accepted limit only moves up: after each flight, it is the largest of the design limit and every deviation so far. A smaller deviation leaves it where it is.",
            "Start `limit` at `design` and `accepted` as an empty list. In a `for` loop over `deviations`, ask whether the `deviation` is bigger than `limit`; if so, it becomes the new `limit`. Then add `limit` to `accepted`, and return `accepted` after the loop.",
            "`        if deviation ___ limit:`",
        ],
        why="`limit = design` and `accepted = []`, then `for deviation in deviations:` with `if deviation > limit:` and `limit = deviation`, and after that `accepted.append(limit)`; after the loop `return accepted`. In the first case, the limit goes from 1 to 1.4, 2.0 and 3.1.",
    )
    return


@app.cell(hide_code=True)
def text_after_stretch():
    mo.md(r"""
    In the first case, the design said 1, and after six flights a deviation of 3.1 counts as acceptable, more than three times the design. Each step was measured against the last accepted limit, not against the design: the limit grew by 0.4, 0.6 and 1.1, and each time nothing went wrong.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_mean():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. running_mean: the mean so far

        Write `running_mean(values)`. It returns a list `means` with the mean of the first value, of the first two, of the first three, and so on. Keep the sum so far in a variable `total` and the number of values so far in `count`, and go through the values with a `for` loop, with `value` for each one. The list can be empty. Replace the `...`. The check calls `running_mean([2, 4, 6])`, then other lists.
        """),
        L.task("The mean after each value."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mean():
    box_mean = L.code_box("running_mean", lines=9)
    box_mean
    return (box_mean,)


@app.cell(hide_code=True)
def check_mean(box_mean):
    result_mean = L.check("running_mean", box_mean.value)
    result_mean
    return (result_mean,)


@app.cell(hide_code=True)
def hints_mean():
    L.hints(
        "running_mean",
        [
            "The mean so far needs two numbers: the sum so far and how many values there have been so far. Both grow by one value in each step of the loop.",
            "Start `total` and `count` at 0 and `means` as an empty list. In the loop, add the `value` to `total` and 1 to `count`, then add the mean so far to `means`. Return `means` after the loop.",
            "`        means.append(total ___ count)`",
        ],
        why="`total = 0`, `count = 0` and `means = []`, then `for value in values:` with `total = total + value`, `count = count + 1` and `means.append(total / count)`; after the loop `return means`. For `[2, 4, 6]` that gives `[2.0, 3.0, 4.0]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_mean():
    mo.md(r"""
    In the case `[1, 1, 1, 97]`, the mean is 1 for three values and 25 after the fourth: one value moves it more than all the others together. The median of those four values is still 1.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: a thin tail and a long tail

    Heights and incident sizes made the same way, drawn with a fixed seed, up to 1,000,000 of each. Choose how many draws to look at, and compare the mean, the median and the largest value of both, and the share of the total that the largest value takes. The chart shows the mean of the incident sizes after each draw.
    """)
    return


@app.cell(hide_code=True)
def demo_tails():
    tails = L.explore("draws", "draws of each", lambda _v: C.demo_frame(_v, L.lang),
                      start=1000, values=C.DRAWS, role="",
                      task="Find the smallest number of draws on the slider at which one single incident makes up more than a fifth of the total of all incident sizes.")
    tails
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    At 10,000 draws: draw 6,419 is an incident of about 10,500, about a quarter of the total of all 10,000. That one draw lifts the mean from 2.85 to 4.49 in a single step; the median barely moves. The heights never do this: at 1,000 draws their tallest is about a thousandth of their total, and the share only gets smaller after that. With 1,000,000 draws the mean of the incidents is back near 3, and that one incident is still the largest. Someone who had looked only at the first 6,418 draws would never have seen an incident above 263, about a fortieth of it.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_drift():
    mo.md(r"""
    ### Exercise 3. Fewer tests every month

    A team releases a new version of its chatbot every month, and runs a set of safety tests before each release. To save time, it drops a few tests each month that have never failed. After a year, half of the tests are gone, and not one release has caused an incident.
    """)
    return


@app.cell(hide_code=True)
def choice_drift_cell():
    choice_drift = L.choice(
        "drift",
        question="What does this lesson say about the team's reasoning?",
        options={
            "fine": "It is sound: a test that has not failed in a year protects against nothing",
            "right": "A year without incidents does not show that the remaining tests are enough",
            "swan": "Whatever harm comes later is a black swan, so the team could not have done any better",
        },
        correct="right",
        why={
            "right": "Each cut is accepted because the last one did no harm: that is drift into failure. Feynman's words about the Shuttle fit here too: the success of the flights is taken as evidence of safety. Under drift, the risk grows while nothing happens.",
            "fine": "A test that never failed may not have met its problem yet. A year without an incident is what drift looks like from the inside: each step looks fine until the accident. An absence of evidence is not strong evidence of absence. See *Drifting, one small step at a time* in *The idea*.",
            "swan": "A black swan is an event that most people did not see coming. Here the team chose each step, and a risk that grows step by step can be seen and stopped. See *Drifting, one small step at a time* in *The idea*.",
        },
    )
    choice_drift
    return (choice_drift,)


@app.cell(hide_code=True)
def text_exercise_whose_swan():
    mo.md(r"""
    ### Exercise 4. Whose black swan?

    In March, a red team, people who attack a system on purpose to find its weak spots, writes in a report that a new kind of jailbreak, a trick that gets chatbots past their safeguards, could appear and spread fast. In June it appears and is used against many chatbots at once. Most of the companies that run them had never heard of it.
    """)
    return


@app.cell(hide_code=True)
def choice_whose_swan_cell():
    choice_whose_swan = L.choice(
        "whose_swan",
        question="Is the jailbreak a black swan?",
        options={
            "right": "For most companies yes; for the red team it was a foreseen tail event",
            "no": "No: an event is only a black swan if nobody anywhere could have foreseen it at all",
            "yes": "Yes, for everyone: it was rare, had a large impact, and came without warning",
        },
        correct="right",
        why={
            "right": "A black swan depends on who is looking, as Taleb's turkey and butcher show. Under the textbook's working definition, an event that was largely unexpected for most people counts, while those who foresaw it saw a tail event.",
            "no": "That is the strict definition, which the textbook does not use. Under its working definition, an event that was largely unexpected for most people counts as a black swan. See *Tail events and black swans* in *The idea*.",
            "yes": "The red team did see it coming, so for them it came with a warning. Black swans depend on who is looking. See *Tail events and black swans* in *The idea*.",
        },
    )
    choice_whose_swan
    return (choice_whose_swan,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Safeguards in a row](/course/open/rai-swiss-cheese?lang=en) ended with what its models leave out, and [Random numbers and seeds](/course/open/np-random?lang=en) gave the generator that drew this lesson's tails. This lesson added drift into failure, with Columbia's foam as a documented case, and tail events, black swans and tail risk, where one draw can outweigh thousands. The next lesson, *Measuring toxicity*, turns to a safeguard that many chatbots use: a model that scores how harmful a text is.

    **An everyday example.** A bridge that creaks a little more every winter, and is used every day without trouble, has not shown that it is safe: each winter without a collapse is another step of the same drift.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_drift, choice_whose_swan, result_mean, result_stretch):
    L.wrap_up(result_stretch, result_mean, choice_drift, choice_whose_swan)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
