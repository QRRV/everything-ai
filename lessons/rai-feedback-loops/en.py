# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-feedback-loops. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-feedback-loops", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Bias at every step](/course/open/rai-bias-lifecycle?lang=en) and [Random numbers and seeds](/course/open/np-random?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_measure():
    quick_measure = L.choice(
        "quick-measure",
        question="A model uses arrests as its label for crime, while some neighborhoods are policed far more than others. Which kind of bias is that?",
        options={
            "historical": "Historical bias",
            "right": "Measurement bias",
            "deployment": "Deployment bias, as the police use it",
        },
        correct="right",
        why={
            "historical": "The world may be unequal too, but here the label itself measures crime worse in some places. See *Making the data* in [Bias at every step](/course/open/rai-bias-lifecycle?lang=en).",
            "right": "An arrest is a stand-in for crime, and it measures crime worse where there are more police.",
            "deployment": "Deployment bias is about using a model for something it was not built for. See *Making the data* in [Bias at every step](/course/open/rai-bias-lifecycle?lang=en).",
        },
    )
    quick_measure
    return (quick_measure,)


@app.cell(hide_code=True)
def quick_check_seed():
    quick_seed = L.choice(
        "quick-seed",
        question="You run `np.random.default_rng(7).random(size=3)` twice. What do you get?",
        options={
            "different": "Two different sets of three numbers",
            "right": "The same three numbers twice",
            "sevens": "Three numbers close to 7",
        },
        correct="right",
        why={
            "different": "With the same seed, a generator gives the same numbers every time. See *The idea* in [Random numbers and seeds](/course/open/np-random?lang=en).",
            "right": "The seed fixes the numbers: anyone who runs this gets the same three.",
            "sevens": "The seed is not a value to aim for: `rng.random` gives numbers from 0 up to 1. See *The idea* in [Random numbers and seeds](/course/open/np-random?lang=en).",
        },
    )
    quick_seed
    return (quick_seed,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model decides where police patrol. Where they patrol, they see more crime, and what they see goes into the data the model learns from next. So the model seems right, and sends even more police there. In [Where labels come from](/course/open/data-labels?lang=en) the feedback loop was the way from the predictions of a model back into its training data. Here you see that this loop can grow a difference by itself, even between two districts with exactly the same crime.

    **An everyday example.** You look for your keys only in your coat pocket, because that is where you found them last time. You find them there more and more often, partly because you never look anywhere else. Your memory of where the keys turned up says coat pocket, but it only shows where you looked.
    """)
    return


@app.cell(hide_code=True)
def predict_winner():
    guess_winner = L.guess(
        "winner",
        question="Two districts. On a day police are there, they find a crime with a chance of 10 percent in district A and 11 percent in district B. Each day police go to a district with a chance equal to its share of the crimes found so far. In the long run, what share of the days do you think goes to district B?",
        options={
            "fair": "About 52 percent, a little more than half",
            "right": "All of them, in the end: 100 percent",
            "same": "Half, as the difference is tiny",
        },
        reveal="**All of them.** Ensign and colleagues (2018) showed with a model of this loop that, with rates of 10 and 11 percent, it will settle on district B with probability 1: a classic go-with-the-winner problem. They even assumed that police find crime at exactly its true rate. With the same rate in both districts it is no better: where the share ends up is a matter of chance. The demo below shows that.",
    )
    guess_winner
    return (guess_winner,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **A prediction that makes itself true.** In 1948 Robert K. Merton named the self-fulfilling prophecy: "a false definition of the situation evoking a new behavior which makes the originally false conception come true." In his parable, set in 1932, a sound bank collapses because its customers believe a rumor that it is in trouble, and all come to take their money out. It only happens among people, he added: predictions of the return of Halley's comet do not change its orbit. The book *Fairness and Machine Learning* describes something similar for models, a self-fulfilling prediction: police sent to the areas a model calls high-risk find more crime there, so "the prediction will appear to be validated, even if it had been made purely based on data biases." The arrests from those areas go back into the training data, and the gap can grow over time.

    **Oakland.** Kristian Lum and William Isaac (2016) ran the algorithm of PredPol, a large US vendor of such systems, on the drug arrests of the Oakland police in 2010. The algorithm uses only three things: the type, place and time of past crimes. Drug use, estimated from a national survey, was spread fairly evenly over the city, but the arrests were concentrated in two areas with mostly non-white, low-income residents, with about 200 times more drug arrests than elsewhere. In their simulation, Black people would be targeted at roughly twice the rate of White people. When they let each targeted place record 20 percent more crime, the model grew ever more confident that the crime was where it had sent the police. It was a simulation, not a report on what the Oakland police did.

    **Found or reported.** Ensign and colleagues (2018) split police data in two. What police discover where the model sent them feeds the loop. What people report, such as a call to the emergency number, comes in whether or not the model sent anyone: it can dampen a loop, but not remove it entirely. And they explain why a loop runs away: "if our decision procedure sends police to region A 90% of the time, we should not be surprised that discovered incidents in region A happen at a rate of nine to one, even if the crime rate is the same across both regions." Their fix keeps a discovery only with a chance that falls the more often police were sent there. A plain version of the same idea: divide what was found by how often someone looked.

    **Try it.** The cell below runs the loop of Ensign and colleagues five times, each with another seed. Two districts have the same crime: on a day police are in one, they find a crime there with a chance of 0.3. Each day police go to district 0 with a chance equal to its share of the crimes found so far. The cell prints the seed, the crimes found in district 0 and 1 after 1000 days, and the share of district 0. Each count starts at 1, so that both districts have a chance on the first day. Run the cell.
    """)
    return


@app.cell
def demo_loop():
    import numpy as np

    # two districts with the same crime: on a day with police, they find a crime there with chance 0.3
    _rate = 0.3
    for _seed in [1, 2, 3, 4, 5]:
        _rng = np.random.default_rng(_seed)
        _found = [1, 1]   # crimes found so far in district 0 and 1
        for _day in range(1000):
            # police go to district 0 with a chance equal to its share of the crimes found so far
            _share = _found[0] / (_found[0] + _found[1])
            if _rng.random() < _share:
                _sent = 0
            else:
                _sent = 1
            if _rng.random() < _rate:
                _found[_sent] = _found[_sent] + 1
        print(_seed, _found, round(_found[0] / (_found[0] + _found[1]), 2))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.picture({"title": "Five runs of the loop", "xlabel": "day", "ylabel": "share found in district 0",
                           "true": "share of the crime: 0.5", "decimal": ".",
                           "alt": "Five orange lines of the share of the crimes found that are in district 0, over 1000 days. After 200 days each stays in a narrow band of its own; at day 1000 they are at {finals}. A dashed line at 0.5 is the share of the crime that is really in district 0."})),
        mo.md(r"""
        After 200 days, each run stays in a narrow band of its own, far from a half, while the crime is split half and half. What police find grows with where they are sent, so the counts copy the split instead of the crime.

        **In Europe and the Netherlands.** The EU Agency for Fundamental Rights (FRA) simulated two districts with the same crime in 2022, starting with 20 and 80 percent of the patrols. Sharing the patrols in proportion to the crime recorded kept that split for ever; a simple learning model, in one run, sent all police to one district after 40 weeks. "These results show that feedback loops can easily occur in fully automated settings," the FRA wrote. The Dutch police used the Crime Anticipation System (CAS) from the mid-2010s until December 2025. It predicted crime in squares of 125 by 125 meters, from reports and calls by citizens and from known suspects living nearby; in the terms of Ensign and colleagues, reports dampen a loop. Amnesty International (2020) described a loop in another Dutch system, around Roermond: a car that the system flags and that police follow up stays linked to the profile in police databases, even when the stop finds nothing, so the next time it drives there, the system rates it as a higher risk.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **self-fulfilling prediction** | a prediction that changes what happens or what gets measured, so that the data seems to prove it right |
        | **discovered incident** | a crime that police found where a system sent them; it feeds the loop |
        | **reported incident** | a crime that people report themselves, such as with a call; it comes in without the system |
        | **runaway feedback loop** | a feedback loop that keeps growing a difference, until nearly everything goes to one place |
        """),
        L.watch_out("A feedback loop does not need biased data to start. Ensign and colleagues assumed that police find crime at exactly its true rate, and their loop still ran away. A biased start, such as the arrests in Oakland, only makes it worse."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To look for a feedback loop:

    1. Follow the decisions of a model: what do people do with them?
    2. Ask whether that changes the data that comes back: who gets looked at, who gets a decision and so an outcome, and who never does.
    3. Split the data into what was found because of the model, and what came in without it.
    4. Divide what was found by how often someone looked: per day with police, per check, per person hired.
    5. If a difference grows while the world did not change, do not trust the totals. Count discoveries in proportion to the effort, and use data that does not depend on the model, knowing that it dampens a loop but does not remove it.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The records of one year for two districts, made up by the course. `_found` holds the crimes police found in each. The loop works out each district's share of all the crimes found, as a report or a model would see it. Run the cell.
    """)
    return


@app.cell
def example_share():
    # made-up records of one year: crimes the police found, per district
    _found = {"North": 99, "South": 11}
    _total = _found["North"] + _found["South"]

    _share = {}
    for _district in _found:
        _share[_district] = round(_found[_district] / _total, 2)
    print(_share)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    North has 90 percent of the crimes found: nine times as many as South. A model that learns from this sends even more police to North. But how often were the police there? Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_per_day():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Per day with police

        `days` holds how many days of the year the police were in each district, and `found` the crimes they found there. Replace the `...` with a loop over the districts, as in the worked example, with the names without their underscore: `per_day` and `district`. Put in `per_day`, for each district, the crimes found divided by the days with police, rounded to two decimals. The last line is `per_day`, so the check sees it. The check also tries other numbers.
        """),
        L.task("Work out `per_day`: for each district, the crimes found per day with police."),
    ])
    return


@app.cell(hide_code=True)
def exercise_per_day():
    box_per_day = L.code_box("per-day", lines=10)
    box_per_day
    return (box_per_day,)


@app.cell(hide_code=True)
def check_per_day(box_per_day):
    result_per_day = L.check("per-day", box_per_day.value)
    result_per_day
    return (result_per_day,)


@app.cell(hide_code=True)
def hints_per_day():
    L.hints(
        "per-day",
        [
            "Use the loop of the worked example. Only the division changes: not by the total of all crimes found, but by the days the police were in that district.",
            "The days the police were in a district are in `days`, under the name of the district: `days[district]`.",
            "`per_day[district] = round(found[district] / ___, 2)`.",
        ],
        why="`per_day = {}`, then `for district in found:` with `per_day[district] = round(found[district] / days[district], 2)` under it. That gives `{'North': 0.3, 'South': 0.31}`: 99 crimes in 330 days, and 11 in 35.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_sources():
    mo.md(r"""
    ### Exercise 2. Found or reported?

    Six ways a crime comes to the police. Sort each: found because a system sent someone, or reported by people themselves.
    """)
    return


@app.cell(hide_code=True)
def sort_sources_cell():
    sort_sources = L.sort("sources")
    sort_sources
    return (sort_sources,)


@app.cell(hide_code=True)
def check_sources(sort_sources):
    result_sources = L.check("sources", sort_sources)
    result_sources
    return (result_sources,)


@app.cell(hide_code=True)
def hints_sources():
    L.hints(
        "sources",
        [
            "Ask of each case: would this crime have reached the police if no system had sent anyone?",
            "The call and the report at the station come in without a system. Which case, besides the patrol and the stop, was found because a system pointed at it?",
            "Found because the system sent someone: on patrol, a stop and \\_\\_\\_. Reported by people themselves: \\_\\_\\_, at the station and online.",
        ],
        why="**Found because the system sent someone:** on patrol, a stop and a flagged car. **Reported by people themselves:** the emergency call, at the station and online.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_hiring():
    mo.md(r"""
    ### Exercise 3. Outside the police
    """)
    return


@app.cell(hide_code=True)
def choice_hiring_cell():
    choice_hiring = L.choice(
        "hiring",
        question="A hiring model learns only from how the people it hired did at work. Over the years it hires fewer and fewer people from group X. Is this a feedback loop?",
        options={
            "no": "No, nobody is arrested here",
            "right": "Yes, its choices decide what it learns",
            "named": "Only if group X is a column in the data",
        },
        correct="right",
        why={
            "no": "A feedback loop needs no police: Ensign and colleagues name hiring, loans and college admissions too. See *Found or reported* in *The idea*.",
            "right": "It only sees how the people it hired did. The people from group X it rejected never get an outcome, so it can never learn that it was wrong about them.",
            "named": "The loop needs no column: it runs through who gets a decision, and so an outcome. See *Oakland* in *The idea*.",
        },
    )
    choice_hiring
    return (choice_hiring,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Bias at every step](/course/open/rai-bias-lifecycle?lang=en) a dashed arrow ran from the use of a model back to the world; here you saw what that arrow can do. The decisions of a model change what gets measured next, so even two districts with the same crime drift apart, and the data seems to prove the model right. In [Where labels come from](/course/open/data-labels?lang=en) you saw that a model learns only about what earlier choices let through; here you saw how far that loop can go. Your division by the days with police is a first tool against it. The next lesson, *Leaving out the column is not enough*, shows another way a model can treat groups differently without being told who is in which group.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_hiring, result_per_day, result_sources):
    L.wrap_up(result_per_day, result_sources, choice_hiring)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
