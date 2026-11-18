# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-case-uber-crash. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-case-uber-crash", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A person in the loop](/course/open/rai-human-oversight?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bias():
    quick_bias = L.choice(
        "quick-bias",
        question="A system flags nothing, and the person checking it misses a problem that their own information showed. What is that called?",
        options={
            "gap": "A responsibility gap, as nobody answers for it",
            "right": "Automation bias",
            "fine": "Oversight that works as it should, as the system decides",
        },
        correct="right",
        why={
            "gap": "A responsibility gap is about who answers for what a system did. See *The idea* in [A person in the loop](/course/open/rai-human-oversight?lang=en).",
            "right": "Missing what the system did not flag is one of the two forms of automation bias; following its wrong suggestions is the other.",
            "fine": "The person was there to catch what the system missed. See *The idea* in [A person in the loop](/course/open/rai-human-oversight?lang=en).",
        },
    )
    quick_bias
    return (quick_bias,)


@app.cell(hide_code=True)
def quick_check_three():
    quick_three = L.choice(
        "quick-three",
        question="Oversight works only when the person has three things. Which three?",
        options={
            "course": "A course about the law, a manual, and a manager who checks the work",
            "right": "Time, the case itself, and the power to say no",
            "trust": "Trust in the system, and speed",
        },
        correct="right",
        why={
            "course": "Those may help, but they do not give the person room to catch a mistake. See *The idea* in [A person in the loop](/course/open/rai-human-oversight?lang=en).",
            "right": "Time to look, information about the case itself and not only a score, and the power to say no without paying for it.",
            "trust": "Trust in the system is what automation bias is made of. See *The idea* in [A person in the loop](/course/open/rai-human-oversight?lang=en).",
        },
    )
    quick_three
    return (quick_three,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In March 2018 a test car drove itself through Tempe, in Arizona, at night, with one person at the wheel who was there to watch. It hit and killed a pedestrian who was walking a bicycle across the road. The car's system had detected the pedestrian seconds before; the person at the wheel was looking at a phone. The investigators found not one failure but a row of them, and several were choices in the design.

    **An everyday example.** Hold a few slices of cheese with holes in front of a lamp. Light gets through only where there is a hole in every slice at the same spot. Each slice is a layer of safety; an accident needs a hole in all of them at once.
    """)
    return


@app.cell(hide_code=True)
def predict_detected():
    guess_detected = L.guess(
        "detected",
        question="The car drove at about 44 miles an hour, at night. How long before the crash do you think its system first detected the pedestrian?",
        options={
            "half": "About half a second before",
            "one": "About one and a half seconds",
            "right": "5.6 seconds before",
        },
        reveal="**5.6 seconds before**, first by radar. But it never recognized a pedestrian, and the car began to slow down only 0.2 seconds before the crash.",
    )
    guess_detected
    return (guess_detected,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The crash.** On Sunday 18 March 2018, at 9:58 in the evening, a test car of Uber's Advanced Technologies Group drove itself along Mill Avenue in Tempe: a Volvo XC90 with Uber's automated driving system, still in development, which had been driving itself for about 19 minutes. Elaine Herzberg, 49, was walking a bicycle across the road, at a place without a crosswalk. The car came at 45 miles an hour, the speed limit, and struck the pedestrian at 39. Herzberg was killed. The NTSB, the American board that investigates transport accidents, published its report in November 2019.

        **What the system saw.** The system detected the pedestrian 5.6 seconds before impact, first by radar, as a vehicle. After that it took the pedestrian for *other*, a vehicle again, switched between those two several times, then took the pedestrian for a bicycle, *other*, and at 1.2 seconds a bicycle again. Each time the kind changed, the system no longer used the tracking history of the object, so it could not correctly predict the path; and an object of kind *other* was treated as standing still. It never recognized a pedestrian: the design did not count on pedestrians crossing outside a crosswalk.

        **What the system did.** At 1.2 seconds the system saw an emergency. Then came *action suppression*: for 1 second the system held back braking, while it checked the hazard or the operator took over, and it did not warn the operator. Uber had built that in because of concerns about false alarms. The system braked at most 7 m/s², and only when that would avoid a crash, not to make a crash less severe. 0.2 seconds before the crash a sound went off and the car began to slow down. Volvo's own emergency braking, which could detect pedestrians, was switched off whenever Uber's system drove: the radars of the two used the same frequencies, and the brakes could not choose between two commands. In a simulation by Volvo, with ideal sensors, that system avoided the crash in 17 of 20 variations of the pedestrian's movement, and hit at under 10 miles an hour in the other 3.

        **The person in the car.** One operator sat in the driver's seat; until the autumn of 2017 Uber had put two people in each car. The operator was streaming a television show on a phone, looked down 34 percent of the time the car was moving, and looked down from 6 seconds before the crash for 5 seconds. Uber rarely checked the camera images of its operators.

        **The verdict.** The NTSB named as the probable cause that the operator did not watch the road and the system, distracted by a phone. Contributing were three failings of Uber's group: inadequate procedures for assessing safety risks, ineffective oversight of its operators, and no adequate way to keep operators from relying on the system too much, "all a consequence of its inadequate safety culture". Further factors were the pedestrian's crossing outside a crosswalk, impaired by a drug according to the report, and too little oversight of the testing by Arizona's transport department.

        **Afterwards.** Uber settled with the family before the month was out; the terms were not made public. In March 2019 the Yavapai County Attorney wrote that there was no basis for criminal liability of Uber. The operator was charged with negligent homicide in 2020, pleaded guilty to endangerment in July 2023, and got three years of supervised probation.

        No single failure caused the crash. The system detected the pedestrian but never recognized a person, dropped the path at every change, and held back the brakes; a second system that could brake was switched off. The person who had to catch all that was alone in a car that used to have two, looking down, and rarely checked. Only that person was charged.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **safety driver** | a person in a self-driving test car who watches the road and the system and must take over when something goes wrong; the NTSB calls this person the vehicle operator |
        """),
        L.watch_out("Detected is not recognized. The system saw *something* 5.6 seconds before the crash, but did not know what it was or where it was going. \"The car did not see the pedestrian\" is not what happened."),
        mo.Html(C.picture({"title": "The last six seconds before the crash",
                           "rows": ["What the system took the pedestrian for", "What the system did", "The operator"],
                           "kinds": {"vehicle": "vehicle", "other": "other", "bicycle": "bicycle"},
                           "switches": "back and forth", "held": "emergency: braking held back",
                           "key": "Middle row, outlined: braking held back; filled: a sound, and the car slows down. "
                                  "Bottom row, outlined: the operator looks down; filled: looks at the road.",
                           "down": "looks down", "impact": "crash, 39 mph",
                           "xlabel": "seconds before the crash", "decimal": ".",
                           "alt": "A timeline of the last six seconds. The system detects the pedestrian at 5.6 seconds as a vehicle, then switches between other, vehicle and bicycle. At 1.2 seconds it sees an emergency and holds back braking for 1 second; at 0.2 seconds a sound goes off and the car slows down. The operator looks down from 6 seconds until 1 second before the crash."})),
        mo.md(r"""
        From the NTSB report. Between 3.8 and 2.7 seconds the system switched several more times between vehicle and other.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To look for the layers in an accident with an automated system:

    1. List every layer that could have stopped it: what the system perceives, what it predicts, how it brakes, a second system, the person, the checks of the company, the regulator.
    2. For each layer, write down what it did, and whether that was a choice in the design.
    3. Mark the layers that someone switched off or removed, and why.
    4. Ask how the person in the loop was helped to watch, and how that person was checked.
    5. Compare where the blame landed with where the holes were.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The rule from the report, simplified: when the kind of an object changes, the system forgets its path so far and counts again from that moment. `_kinds` holds an object made up by the course, seen once a second, with the seconds before a crash as keys. The loop goes through the moments with `_time`; when the kind differs from `_previous`, `_start` moves to that moment, and `_previous` remembers each kind. `_history` is how many seconds of path the system still counts at `_now`. -5.0 means 5 seconds before. `_previous` starts as an empty string, so the first moment always counts as a change and gives `_start` a value. Run the cell.
    """)
    return


@app.cell
def example_history():
    _kinds = {-5.0: "bicycle", -4.0: "bicycle", -3.0: "other", -2.0: "other", -1.0: "other"}
    _now = -1.0

    _previous = ""
    for _time in _kinds:
        if _kinds[_time] != _previous:
            _start = _time
        _previous = _kinds[_time]
    _history = _now - _start
    print(_start, _history)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The object was seen for 4 seconds, but at -1.0 the system counts only 2.0 of them: from -3.0, when it became *other*. Now the same rule for the moments of the report: Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_history():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. How much of the path was left?

        The moments from the NTSB report. `kinds` holds the seconds before the crash and what the system took the pedestrian for; the switches between -3.8 and -2.7 are left out, which does not change the answer. `now` is -1.2, the moment the system saw the emergency. Replace the `...` with the lines of the worked example, with the names without their underscore: `previous`, `time`, `start` and `history`. The loop goes through `kinds` with `time`. Leave out the `print`: the last line is `history`, so the check sees it. The check also tries other moments.
        """),
        L.task("Work out `history`: how many seconds of the pedestrian's path the system still counted when it saw the emergency."),
    ])
    return


@app.cell(hide_code=True)
def exercise_history():
    box_history = L.code_box("history", lines=12)
    box_history
    return (box_history,)


@app.cell(hide_code=True)
def check_history(box_history):
    result_history = L.check("history", box_history.value)
    result_history
    return (result_history,)


@app.cell(hide_code=True)
def hints_history():
    L.hints(
        "history",
        [
            "The worked example does this for a made-up object. Copy its lines, without the `print`, and take the underscore off each name.",
            "Start `previous` as `\"\"`. In the loop, move `start` to `time` when the kind differs from `previous`, and then let `previous` remember the kind. After the loop, subtract.",
            "In the loop: `if kinds[time] != ___:` with `start = time` under it; after the loop: `history = now - ___`.",
        ],
        why="`previous = \"\"`, then `for time in kinds:` with `if kinds[time] != previous:` and under it `start = time`, then `previous = kinds[time]` in the loop, and after the loop `history = now - start`. That gives `0.0`: at -1.2 the kind had just changed.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_verdict():
    mo.md(r"""
    ### Exercise 2. The verdict

    Six findings of the NTSB. Sort each one as the NTSB did.
    """)
    return


@app.cell(hide_code=True)
def sort_verdict_cell():
    sort_verdict = L.sort("verdict")
    sort_verdict
    return (sort_verdict,)


@app.cell(hide_code=True)
def check_verdict(sort_verdict):
    result_verdict = L.check("verdict", sort_verdict)
    result_verdict
    return (result_verdict,)


@app.cell(hide_code=True)
def hints_verdict():
    L.hints(
        "verdict",
        [
            "The NTSB named one probable cause, and then factors that contributed: some of Uber's group, some of others.",
            "Uber's group had three: about risks, about the operators, and about relying on the system.",
            "Cause: \\_\\_\\_. Uber: the risks, \\_\\_\\_ and the trust. Further: the crossing and \\_\\_\\_.",
        ],
        why="**The probable cause:** the phone. **Contributing, Uber:** the risks, the oversight and the trust. **Further factors:** the crossing and the state.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_court():
    mo.md(r"""
    ### Exercise 3. What happened in court?
    """)
    return


@app.cell(hide_code=True)
def choice_court_cell():
    choice_court = L.choice(
        "court",
        question="What came of the crash in court?",
        options={
            "uber": "Uber was convicted, and the operator was cleared",
            "right": "The operator pleaded guilty; Uber was not charged",
            "nobody": "Nobody was charged, because the system itself was driving the car",
        },
        correct="right",
        why={
            "uber": "Uber was never charged: in 2019 the Yavapai County Attorney saw no basis for criminal liability of Uber. See *The idea*.",
            "right": "In 2019 the Yavapai County Attorney saw no basis for criminal liability of Uber; in 2023 the operator pleaded guilty to endangerment and got three years of supervised probation. The blame landed on the person nearest the machine: the moral crumple zone of [Who is responsible?](/course/open/rai-accountability?lang=en).",
            "nobody": "The operator was charged with negligent homicide in 2020. See *The idea*.",
        },
    )
    choice_court
    return (choice_court,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [A person in the loop](/course/open/rai-human-oversight?lang=en) you saw that a person who watches a system that is mostly right often misses what it does not flag, and that oversight needs time, information and the power to say no. Here the operator had to watch the whole trip, alone, was not warned when the system held back the brakes, and was rarely checked. In [Who is responsible?](/course/open/rai-accountability?lang=en) you met the moral crumple zone: in court the blame landed on the operator alone. This ends the module *Ethics*; the next module, *Fairness*, starts with *Direct and indirect discrimination*.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_court, result_history, result_verdict):
    L.wrap_up(result_history, result_verdict, choice_court)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
