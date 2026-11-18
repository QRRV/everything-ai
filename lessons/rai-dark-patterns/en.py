# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-dark-patterns. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-dark-patterns", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sum():
    quick_sum = L.choice(
        "quick-sum",
        question="Which theory judges a choice by what it brings about, added up over everyone?",
        options={
            "right": "Consequentialism",
            "deontology": "Deontology, with its duties and rights",
            "care": "Care ethics, from relationships",
        },
        correct="right",
        why={
            "right": "Consequentialism adds up the good and the bad that each option brings about.",
            "deontology": "Deontology asks which duties and rights hold, whatever the outcome. See *The idea* in [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).",
            "care": "Care ethics starts from who depends on whom. See *The idea* in [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).",
        },
    )
    quick_sum
    return (quick_sum,)


@app.cell(hide_code=True)
def quick_check_means():
    quick_means = L.choice(
        "quick-means",
        question="Immanuel Kant held that you may never treat a person merely as a means. Which theory is that?",
        options={
            "virtue": "Virtue ethics",
            "right": "Deontology",
            "veil": "A fair agreement behind a veil",
        },
        correct="right",
        why={
            "virtue": "Virtue ethics asks what a person of good character would do. See *The idea* in [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).",
            "right": "A duty that holds whatever the outcome: deontology.",
            "veil": "The veil of ignorance is John Rawls's idea of a rule that everyone could accept. See *The idea* in [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en).",
        },
    )
    quick_means
    return (quick_means,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Every app and website is designed to steer what you do: which button is big, what is chosen for you, how many steps it takes to say no. AI makes that steering personal: a system can learn what works on you, and change what it shows you from moment to moment. This lesson asks when steering is fine, when it works against people, and when it becomes manipulation.

    **An everyday example.** A supermarket puts sweets at the checkout, at the height of a child's eyes.
    """)
    return


@app.cell(hide_code=True)
def predict_share():
    guess_share = L.guess(
        "share",
        question="In 2019, researchers searched about 11,000 shopping websites for tricks such as fake countdown timers. On how many of them do you think they found at least one?",
        options={
            "few": "About 1 in 100",
            "right": "About 1 in 10",
            "half": "About 1 in 2",
        },
        reveal="**About 1 in 10**: 1,818 tricks on 1,254 websites, such as timers that start again, messages that only a few items are left, and products added to the cart without asking (Mathur and colleagues, 2019). Their crawler could only find tricks that show in the text of a page, so the real number is likely higher.",
    )
    guess_share
    return (guess_share,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **nudge**, a word from Richard Thaler and Cass Sunstein (2008), is a change in how a choice is presented that steers people in a predictable direction, without forbidding anything or changing the costs much: an order, a reminder, a default. Defaults are strong. In countries where you are an organ donor unless you say no, far more people count as donors than in countries where you have to say yes (Johnson and Goldstein, 2003). Since 2020 the Netherlands registers everyone aged 18 or older who does not choose as having no objection.

        A **dark pattern**, a name from the designer Harry Brignull (2010), is a design that gets people to do what they did not mean to, against their own interest: a box ticked in advance, a cost that appears at the last step, a hurry that is not real, a subscription that is easy to start and hard to stop.

        A **hypernudge**, a word from the legal scholar Karen Yeung (2017), is a nudge driven by data about you, which adapts all the time and to each person: a feed that changes what it shows from what kept you scrolling. Others may not see the same thing, which makes it hard to notice and hard to resist.

        Steering becomes *manipulation* when it works by going around a person's own reasons: when it works in a hidden way on a weak spot, such as tiredness or fear of missing out (Susser, Roessler and Nissenbaum, 2019). It harms **human autonomy**, choosing for your own reasons, and **human dignity**, the worth of each person, who should never be treated merely as a means. The EU's Digital Services Act forbids online platforms to design their interfaces to deceive or manipulate people, and the AI Act bans AI systems that use manipulative techniques to distort people's behavior in a way that causes, or is likely to cause, significant harm.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **nudge** | a small change in how a choice is presented, such as a default option, that steers people in one direction without forbidding anything or changing the costs |
        | **dark pattern** | a design trick in an app or website that pushes people into choices they did not intend, such as a hidden cancel button or a box that is ticked in advance |
        | **hypernudge** | a nudge driven by data about a person, which a system adapts to that person all the time, such as a feed that changes from what kept them scrolling |
        | **human autonomy** | a person's ability to make their own choices for their own reasons |
        | **human dignity** | the worth every person has simply by being human, so that nobody may be treated as a mere object or number |
        """),
        L.watch_out("A nudge is not wrong because it steers: every design steers, because something has to come first. Ask in whose interest it steers, whether people can see it, and how easily they can choose otherwise."),
        mo.Html(C.picture({"title": "Steps to join and to leave five subscriptions", "x": "steps", "harder": "harder to leave",
                           "alt": "Five made-up subscriptions, with the steps to join as an open dot and the steps to leave as a filled dot. Music: 2 and 2. News: 3 and 9. Fitness: 2 and 7. Cloud: 4 and 3. Games: 1 and 6. News, fitness and games are harder to leave than to join."})),
        mo.md(r"""
        Made up by the course. An open dot is how many steps it takes to join, a filled dot how many to leave again.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To check a design for dark patterns:

    1. Write down the choice the design steers, and in whose interest.
    2. Count the steps for each option, such as yes and no, or join and leave.
    3. Look for what is hidden, ticked in advance, or rushed.
    4. Ask whether the design adapts to each person, and from which data.
    5. Make the option that is good for the person at least as easy as the one that is good for the company.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Two cookie banners, made up by the course. `_accept` holds how many clicks it takes to say yes on each, `_reject` how many to say no. The loop goes through the banners and prints the ones where saying no takes more clicks. Run the cell.
    """)
    return


@app.cell
def example_banners():
    _accept = {"banner A": 1, "banner B": 1}
    _reject = {"banner A": 1, "banner B": 4}

    for _banner in _accept:
        if _reject[_banner] > _accept[_banner]:
            print(_banner, "makes saying no harder")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Only banner B is printed: saying no takes four clicks there, and saying yes one. A few extra clicks look small, but over millions of visitors they decide what many people agree to. Exercise 1 does step 2 of the recipe for the subscriptions in the picture.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_harder():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Which ones are hard to leave?

        The subscriptions of the picture. `join` holds how many steps it takes to join each one, `leave` how many to leave again, and `harder` starts as an empty list. The loop goes through the subscriptions with `service`. Replace the `...` with two lines, as in the worked example, but add the name to `harder` instead of printing it. The last line is `harder`, so the check sees it. The check also tries other steps.
        """),
        L.task("Make `harder` list the subscriptions that take more steps to leave than to join."),
    ])
    return


@app.cell(hide_code=True)
def exercise_harder():
    box_harder = L.code_box("harder", lines=9)
    box_harder
    return (box_harder,)


@app.cell(hide_code=True)
def check_harder(box_harder):
    result_harder = L.check("harder", box_harder.value)
    result_harder
    return (result_harder,)


@app.cell(hide_code=True)
def hints_harder():
    L.hints(
        "harder",
        [
            "Step 2 of the recipe: for each subscription, compare its steps to leave with its steps to join, as the worked example compares the clicks.",
            "Which comparison says *more steps to leave than to join*? And how do you add a name to the end of a list?",
            "`if leave[service] ___ join[service]:`",
        ],
        why="`if leave[service] > join[service]:` with `harder.append(service)` under it gives `['news', 'fitness', 'games']`: three of the five are harder to leave, and `music`, with 2 steps both ways, is not.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Exercise 2. Nudge, dark pattern or hypernudge?

    Six designs, made up by the course. Put each one with the kind it belongs to.
    """)
    return


@app.cell(hide_code=True)
def sort_kinds_cell():
    sort_kinds = L.sort("kinds")
    sort_kinds
    return (sort_kinds,)


@app.cell(hide_code=True)
def check_kinds(sort_kinds):
    result_kinds = L.check("kinds", sort_kinds)
    result_kinds
    return (result_kinds,)


@app.cell(hide_code=True)
def hints_kinds():
    L.hints(
        "kinds",
        [
            "Ask for each design: does it steer openly, does it work against the person, or does it adapt to each person from data?",
            "Two designs hide or rush something. Two others learn something about each person.",
            "Nudge: the salad and \\_\\_\\_. Dark pattern: \\_\\_\\_ and the timer. Hypernudge: \\_\\_\\_ and \\_\\_\\_.",
        ],
        why="**Nudge:** the salad and the pension. **Dark pattern:** the ticked box and the timer. **Hypernudge:** the feed and the playlist.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_manipulation():
    mo.md(r"""
    ### Exercise 3. When does it become manipulation?
    """)
    return


@app.cell(hide_code=True)
def choice_manipulation_cell():
    choice_manipulation = L.choice(
        "manipulation",
        question="A game shows a \"last chance\" offer exactly when its data says that a player is tired and has just lost. What makes this manipulation rather than a nudge?",
        options={
            "price": "The offer costs money, and the player may regret buying it later",
            "game": "It happens inside a game, where people go to relax and not to shop",
            "right": "It uses a weak moment instead of the player's own reasons",
        },
        correct="right",
        why={
            "price": "Many fair offers cost money. What matters is how the offer gets the player to say yes. See *The idea*.",
            "game": "The place is not the problem: the same offer at a fair moment could be fine. See *The idea*.",
            "right": "It is timed, without the player knowing, for a moment when the player can least weigh it: it works on a weak spot and goes around their reasons, which harms their autonomy.",
        },
    )
    choice_manipulation
    return (choice_manipulation,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Ethical theories applied to AI](/course/open/rai-ethical-theories?lang=en) you met the duty never to treat a person merely as a means. A dark pattern does exactly that: it treats people as a source of clicks, not as people with their own reasons. AI adds the hypernudge, steering that adapts to each person and is hard to see. In the next lesson, *Codes of conduct and values in research*, you look at what the people who build such systems owe the public.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_manipulation, result_harder, result_kinds):
    L.wrap_up(result_harder, result_kinds, choice_manipulation)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
