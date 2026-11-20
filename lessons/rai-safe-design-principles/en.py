# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-safe-design-principles. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-safe-design-principles", "en")
    C = L.checks   # checkers that vote on invoices, payments that need a second person, and models that vote


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Nines of reliability](/course/open/rai-nines?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_failures():
    quick_failures = L.choice(
        "quick-failures",
        question="A service has three nines of reliability and handles 100,000 requests a day. About how many requests fail per day?",
        options={
            "right": "100: 0.1% of 100,000",
            "three": "3: one failed request for each nine",
            "percent": "1,000: 1% of 100,000",
        },
        correct="right",
        why={
            "right": "Three nines is 99.9%, so the chance of failure is 0.001: 100,000 × 0.001 = 100 failed requests a day.",
            "three": "The nines count factors of 10 in the chance of failure, not failed requests: three nines is a chance of failure of 0.001, and 100,000 × 0.001 = 100. See [Nines of reliability](/course/open/rai-nines?lang=en).",
            "percent": "1% is the chance of failure at two nines, 99%. Three nines leave 0.1%: 100 of 100,000. See [Nines of reliability](/course/open/rai-nines?lang=en).",
        },
    )
    quick_failures
    return (quick_failures,)


@app.cell(hide_code=True)
def quick_check_steps():
    quick_steps = L.choice(
        "quick-steps",
        question="An AI agent does a task of 3 steps. Each step succeeds with chance 0.9, independently of the others. What is the chance that the whole task succeeds?",
        options={
            "right": "0.729: 0.9 × 0.9 × 0.9",
            "one": "0.9: every step has the same chance",
            "sum": "0.7: one minus the three chances of failure of 0.1",
        },
        correct="right",
        why={
            "right": "All three steps must succeed, and with independent steps the chances multiply: 0.9 × 0.9 × 0.9 = 0.729.",
            "one": "Each step has chance 0.9, but all three must succeed: the chances multiply, 0.9 × 0.9 × 0.9 = 0.729. See [Nines of reliability](/course/open/rai-nines?lang=en).",
            "sum": "Subtracting the chances of failure is not exact: it counts a task in which two steps fail twice. The chances of success multiply: 0.9 × 0.9 × 0.9 = 0.729. See [Nines of reliability](/course/open/rai-nines?lang=en).",
        },
    )
    quick_steps
    return (quick_steps,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A made-up company lets an AI agent handle its invoices. The agent reads each invoice, decides whether it is genuine, and pays it from the company's bank account. It is right 99.9% of the time: three nines. But one wrong decision can send €40,000 to a fraudster, and in this design nothing stands in the way: one model decides, its account can pay any amount, and nobody else looks.

    Engineers met this problem long before AI, in elevators, trains, spacecraft and banks. Their answer was not only to make each part more reliable, but to design the system so that one failing part is not enough to cause harm. This lesson takes their principles and applies them to AI systems: redundancy, fail-safes, separation of duties, least privilege and negative feedback.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="The Space Shuttle flew with five flight computers. How many of them ran the same software?",
        options={
            "all": "All five",
            "right": "Four of the five",
            "none": "None: each had its own",
        },
        reveal="**Four of the five.** During ascent and descent, four computers with identical software formed the redundant set; their commands were voted on, so a failed computer was outvoted. The fifth was a backup with software developed separately, so that one error in the shared software could not stop all five. Redundancy works best when the parts do not fail for the same reason.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Principles for the whole system

    Dan Hendrycks's textbook *Introduction to AI Safety, Ethics, and Society* (2024), which counted reliability in nines in [Nines of reliability](/course/open/rai-nines?lang=en), also gives principles for designing safer systems: "We can reduce both the probability and severity of a system failure by following certain safe design principles when designing it." They belong at the start: "They should be incorporated from the outset, rather than being retrofitted later." And they are about more than the model: "Note that these principles are not only useful in building an AI itself, but also the system around it." The textbook lists eight, among them the five of this lesson. Many are older than AI. In 1975, Jerome H. Saltzer and Michael D. Schroeder listed eight principles for protecting information in computer systems; least privilege is in both lists.

    ### Redundancy

    "Redundancy means having multiple ‘backup’ components", the textbook writes, so that several components can perform each critical function: "In other words, redundancy removes single points of failure." A car with a foot brake and a handbrake can still slow down when one of them fails. That is **redundancy**.

    A well-documented case is the Space Shuttle, described in *Computers in Spaceflight* (1988), a history by James E. Tomayko that NASA published. "Four of the computers, each loaded with identical software, operate in what is termed the ‘redundant set’ during critical mission phases such as ascent and descent." Their commands were voted on at the actuators, the parts that carry out the commands, so that "the failed computer’s commands are physically out-voted". Three is the least for a vote: two computers that disagree give "the feared standoff possible in dual computer systems (one is wrong, but which?)". More computers make a loss rarer: "Triplex computer system failures were expected to cause loss of aircraft three times in a million flights, whereas quadruple computer system failures would cause loss of aircraft only four times in a thousand million flights". Those were projections for aircraft, and in them the fourth computer makes a loss 750 times as rare. During the first of the Shuttle's approach and landing tests, "a computer failed just as the Enterprise was released from the Boeing 747 carrier; yet the landing was still successful."

    But four computers with the same software can make the same mistake at the same moment. The fifth computer was a backup, and "In the interest of avoiding a generic software failure, NASA kept its development separate from PASS", the software of the other four. Redundancy helps when the parts do not fail for the same reason.

    ### Fail-safes

    "Fail-safes are features that aim to ensure a system will be safe even if it fails", in the textbook's words. Its classic example: "Elevator brakes are a classic example of a fail-safe feature. They are attached to the outside of the cabin and are held open only by the tension in the cables that the cabin is suspended on. If tension is lost in the cables, the brakes automatically clamp shut onto the rails in the elevator shaft." In the elevator, the failure itself brings the safe state, without a sensor or a decision in between. That is a **fail-safe**.

    US rules for trains have one too. An alerter in the cab watches whether the engineer who drives the train works the controls. If nothing changes for too long and nobody answers its alarm, that "results in a penalty brake application that brings the locomotive or train to a stop." The time before its alarm sounds, in seconds, is 2400 divided by the track speed in miles per hour: 40 seconds at 60 miles per hour. Before the train leaves, the alerter is tested by letting that time run out.

    For AI, the textbook names possible fail-safes, such as "a kill switch that makes it possible to shut off all instances of an AI system if this is required due to malfunction or other reasons". Saltzer and Schroeder had a related rule for access, fail-safe defaults: "Base access decisions on permission rather than exclusion." Without explicit permission, the answer is no, and a mistake then "tends to fail by refusing permission, a safe situation, since it will be quickly detected."
    """)
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Separation of duties

    "Separation of duties means no single agent can control or misuse the system alone", the textbook writes. Its example is a lab with two chemicals that can explode when mixed in large amounts: each is kept in its own cupboard, and a different person hands out each, in small amounts. "This way, no individual has access to a large amount of both chemicals." For AI it suggests: "We could focus on multiple narrow AI models, instead of a single general one." That is **separation of duties**.

    NIST, the US institute for standards, gives an everyday example: "the person authorizing a paycheck should not also be the one who can prepare them." One form of it is the two-person rule. Saltzer and Schroeder called it separation of privilege: a lock with two keys, so that "no single accident, deception, or breach of trust is sufficient to compromise the protected information." It is used "in bank safe-deposit boxes", and "in the defense system that fires a nuclear weapon only if two different people both give the correct command." In 2025, Anthropic described such a rule for the weights of its AI models: an employee who needs access must "obtain authorization from a second party at the time of the request".

    ### Least privilege

    "Every program and every user of the system should operate using the least set of privileges necessary to complete the job", Saltzer and Schroeder wrote. "Primarily, this principle limits the damage that can result from an accident or error." The textbook adds that separation of duties "might only work if we also ensure individuals do not have access to parts of the system that are not relevant to their tasks". For AI: "Concretely, we might want to avoid plugging AIs into the internet or giving them high-level admin access to confidential information."

    OWASP, a foundation for software security, lists the top 10 risks of applications built on large language models. Its list of 2025 includes AI agents that can do too much, which allows damaging actions "in response to unexpected, ambiguous or manipulated outputs from an LLM, regardless of what is causing the LLM to malfunction". Its example: an extension that only needs to read data connects to a database with an identity that can also change, add and delete data. It asks for the least permissions, for a person to approve actions with a large impact, and for checks outside the model: "Implement authorization in downstream systems rather than relying on an LLM to decide if an action is allowed or not." Logging and limits on how often an agent may act, it adds, do not prevent the risk, but can limit the damage.

    ### Negative feedback

    With positive feedback, a change brings more of the same change: when ice melts, the Earth reflects less sunlight, warms, and more ice melts. "Negative feedback mechanisms act to down-regulate and stabilize systems": a body that gets too warm sweats, and a body that gets cold shivers. For AI, the textbook proposes tools that notice when an AI gets too powerful or behaves hazardously, and then trigger a measure, which "might be a drastic measure like disconnecting an AI from the internet, resetting an AI to a previous version, or using other AIs trained to disempower a powerful AI."

    ### Every principle has a price

    These principles are not rules to apply everywhere. "Note that, depending on the exact type of system, some of these safe design principles might be less useful or even counterproductive", the textbook says, and Saltzer and Schroeder wrote that "they serve best as warnings". Many add parts, cost and complexity. NASA judged the Shuttle's backup worth its weight, power and complexity, as insurance against an error in the shared software. But John R. Garman of NASA's Johnson Space Center said: "we probably did more damage to the system as a whole by putting in the backup". In Tomayko's words, Garman "felt that the institution of the backup took much of the pressure off the developers of the primary system".
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **redundancy** | more than one component for a critical function, so that one failing component is not enough to cause an accident |
        | **fail-safe** | a feature that keeps a system safe when it fails, such as brakes that clamp shut when the cable goes slack |
        | **separation of duties** | splitting a critical task between several people or systems, so that no one can misuse the system alone |
        | least privilege | giving every user and every program only the access their task needs |
        | single point of failure | a part whose failure alone is enough to cause an accident |
        | two-person rule | a form of separation of duties in which two people must act together |
        | majority vote | the answer that more than half of the votes give |
        | negative feedback | a mechanism that counters a change, so that a system stays stable |
        | `not in` | Python's test that something is not in a list: `"pay" not in needs` |
        """),
        L.watch_out("Copies of one part can fail for the same reason: redundancy helps only when the parts do not. And a fail-safe is about what happens when a system fails, not a promise that it never fails."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Find the single points of failure: the parts whose failure alone can cause harm.
    2. Give each critical function more than one part, and make sure those parts do not fail for the same reason.
    3. Decide what the safe state is, and let a failure lead there by itself.
    4. Split critical tasks, so that no one, a person or an AI, can do harm alone.
    5. Give each part only the access its task needs.
    6. Weigh the price: a principle can add parts and cost, and change how people work.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The invoice agent of *Why this matters* needs to read invoices and draft payments. The cell compares that with what its account may do, and prints every right that is more than the task needs.
    """)
    return


@app.cell
def example_privilege():
    needs = ["read invoices", "draft payments"]                  # what the agent's task needs
    has = ["read invoices", "draft payments", "pay", "delete"]   # what its account may do
    for right in has:
        if right not in needs:                                   # more than the task needs
            print("remove:", right)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The account may do two things more than the task needs: pay and delete. With least privilege, both go. If a wrong or manipulated output then asks the agent to pay, its account cannot. In the exercises you build two other principles: a vote of redundant checkers that falls back to the safe side, and a check that needs a second person.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_vote():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. vote: a majority, or hold

        Several checkers, such as models, check each invoice and answer `"pay"` or `"hold"`; a checker that fails gives no answer, `None`. Write `vote(answers)`. It gets the list of `answers` and returns `"pay"` only when more than half of all answers are `"pay"`, counting the ones that are `None` too. Otherwise it returns `"hold"`, the safe answer. Count the pays in a variable `pays`, with a `for` loop over `answers` and `answer` for each answer. Replace the `...`. The check calls `vote(["pay", "pay", "hold"])`, then other lists.
        """),
        L.task("A majority vote that falls back to the safe answer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_vote():
    box_vote = L.code_box("vote", lines=8)
    box_vote
    return (box_vote,)


@app.cell(hide_code=True)
def check_vote(box_vote):
    result_vote = L.check("vote", box_vote.value)
    result_vote
    return (result_vote,)


@app.cell(hide_code=True)
def hints_vote():
    L.hints(
        "vote",
        [
            "A majority vote, as on the Space Shuttle: more than half must agree. When they do not, the function falls back to the safe side, as a fail-safe does.",
            "Start `pays` at 0, and add 1 in a `for` loop for every answer that is `\"pay\"`. After the loop, compare `pays` with half of `len(answers)`: more than half gives `\"pay\"`. Every other case ends at `return \"hold\"`.",
            "`    if pays > ___:`",
        ],
        why="`pays = 0`, a loop that adds 1 for each `\"pay\"`, then `if pays > len(answers) / 2:` with `return \"pay\"`, and after it `return \"hold\"`. For `[\"pay\", None, None]`: one pay of three is not more than half, so hold.",
    )
    return


@app.cell(hide_code=True)
def text_after_vote():
    mo.md(r"""
    Two checkers outvote a third that is wrong, and when two of three have failed, the one that is left cannot pay on its own: the vote holds. That is redundancy and a fail-safe together: a failed checker leads to the safe answer, not to a payment.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pay():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. may_pay: no one approves their own request

        Payments up to and including `LIMIT`, €1,000, may go through without approval. A larger payment needs an approver who is not the one who requested it: the agent may prepare a payment, but not approve it. Write `may_pay(amount, requested_by, approvers)`. It returns `True` when the payment may go through and `False` otherwise; `approvers` is a list of the names of those who approved it, and it may be empty. Go through it with a `for` loop, with `name` for each approver. Replace the `...`. The check calls `may_pay(5000, "agent", ["agent"])`, then other payments.
        """),
        L.task("A check in which no one approves their own payment."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pay():
    box_pay = L.code_box("may_pay", lines=10)
    box_pay
    return (box_pay,)


@app.cell(hide_code=True)
def check_pay(box_pay):
    result_pay = L.check("may_pay", box_pay.value)
    result_pay
    return (result_pay,)


@app.cell(hide_code=True)
def hints_pay():
    L.hints(
        "may_pay",
        [
            "Separation of duties, as in NIST's example of the paycheck: the one who asks is not the one who approves. Up to the limit, there is nothing to check.",
            "First the limit: up to `LIMIT`, return `True`. Above it, go through `approvers`; at a `name` that is not `requested_by`, return `True`. If the loop ends without one, return `False`, after the loop.",
            "`        if name ___ requested_by:`",
        ],
        why="`if amount <= LIMIT:` with `return True`; then `for name in approvers:` with `if name != requested_by:` and `return True` inside it; after the loop, `return False`. The agent's own approval does not count.",
    )
    return


@app.cell(hide_code=True)
def text_after_pay():
    mo.md(r"""
    A request for €5,000 that only the agent approved is blocked; one that Sam approved as well may go. In a real system, such a check runs outside the model, in the payment system, as OWASP asks: then a wrong output of the model cannot switch it off.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: let models vote

    Each model that checks an invoice is wrong with the chance on the slider. The picture shows how often the decision is wrong with one model, with three copies of one model that vote, and with three or five different models that vote and make their mistakes independently of each other, with the curves over all chances up to 60%.
    """)
    return


@app.cell(hide_code=True)
def demo_voting():
    voting = L.explore("error", "chance that one model is wrong, in %", lambda _v: C.demo_frame(_v, L.lang),
                       start=10, values=C.ERRORS, role="",
                       task="Find the smallest chance on the slider at which three different models that vote are wrong more often than one model alone.")
    voting
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    From 60% on. At 50%, voting changes nothing, and above it the majority is wrong more often than one model: voting helps only when each model is right more often than wrong. At 10%, three different models are wrong in 280 of 10,000 invoices and five in 86, against 1,000 for one model, if their mistakes are independent. Copies of one model stay at 1,000: they make the same mistakes, like computers with the same software. If different models share some of their mistakes, the gain lies between those lines.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fail_safe():
    mo.md(r"""
    ### Exercise 3. A robot that fails

    An AI system steers a robot that carries heavy boxes through a warehouse.
    """)
    return


@app.cell(hide_code=True)
def choice_fail_safe_cell():
    choice_fail_safe = L.choice(
        "fail_safe",
        question="Which of these features makes the robot fail-safe?",
        options={
            "backup": "A second controller that takes over when the first one breaks down",
            "right": "Brakes that hold the robot still as soon as it loses power or contact",
            "camera": "A camera that films every move, so that each failure can be studied afterwards",
        },
        correct="right",
        why={
            "right": "A fail-safe lets the failure itself lead to a safe state, like the elevator brakes that clamp shut when the cable goes slack: without power or contact, the robot stands still.",
            "backup": "That is redundancy: it keeps the robot working when one controller fails, but says nothing about what happens when both fail. See *Fail-safes* in *The idea*.",
            "camera": "A camera helps to learn from a failure afterwards; it does not make the robot safe at the moment it fails. See *Fail-safes* in *The idea*.",
        },
    )
    choice_fail_safe
    return (choice_fail_safe,)


@app.cell(hide_code=True)
def text_exercise_privilege():
    mo.md(r"""
    ### Exercise 4. An agent for the calendar

    An AI agent plans meetings in a team's calendars. To do that, it was given an account that can also read and send everyone's email.
    """)
    return


@app.cell(hide_code=True)
def choice_privilege_cell():
    choice_privilege = L.choice(
        "privilege",
        question="What does least privilege ask here?",
        options={
            "approve": "A person who approves every email before the agent sends it",
            "right": "An account for the agent that can reach the calendars and nothing more",
            "log": "A log of every email the agent reads or sends, which someone checks weekly",
        },
        correct="right",
        why={
            "right": "Least privilege gives the agent only the access its task needs: the calendars. Then a wrong or manipulated output cannot read or send email at all.",
            "approve": "An approval limits what the agent does on its own, but the agent can still read everyone's email. Least privilege takes away the access it does not need. See *Least privilege* in *The idea*.",
            "log": "A log helps to find out afterwards what happened; OWASP counts logging among the measures that do not prevent the risk, but can limit the damage. See *Least privilege* in *The idea*.",
        },
    )
    choice_privilege
    return (choice_privilege,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Nines of reliability](/course/open/rai-nines?lang=en) counted how often a system fails. This lesson gave principles that keep a failure from causing harm: redundancy, with parts that do not fail for the same reason; fail-safes, which turn a failure into a safe state; separation of duties and least privilege, which keep one agent, a person or an AI, from doing harm alone; and negative feedback, which counters a change before it grows. The next lesson, *Safeguards in a row*, stacks safeguards and computes the chance that all of them fail at once.

    **An everyday example.** In many companies, the colleague who orders something is not the one who approves the invoice. And many lawn mowers stop their blade when you let go of the handle.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fail_safe, choice_privilege, result_pay, result_vote):
    L.wrap_up(result_vote, result_pay, choice_fail_safe, choice_privilege)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
