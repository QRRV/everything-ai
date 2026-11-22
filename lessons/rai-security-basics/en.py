# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-security-basics. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-security-basics", "en")
    C = L.checks   # threats and an attack tree for the member file of a made-up library


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A function that calls itself](/course/open/py-recursion?lang=en) and [Dictionaries](/course/open/py-dicts?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_recursion():
    quick_recursion = L.choice(
        "quick-recursion",
        question="What does this program show?\n\n```python\ndef f(n):\n    if n == 0:\n        return 1\n    return 2 * f(n - 1)\n\nprint(f(3))\n```",
        options={
            "six": "6",
            "right": "8",
            "never": "Nothing: it never stops",
        },
        correct="right",
        why={
            "right": "`f(3)` gives `2 * f(2)`, `f(2)` gives `2 * f(1)`, `f(1)` gives `2 * f(0)`, and the base case `f(0)` gives 1. On the way back up: 2, 4, 8.",
            "six": "Each call multiplies by 2, three times, and the base case gives 1: 2 × 2 × 2 × 1 = 8. See *Exercise 2. Follow the calls* in [A function that calls itself](/course/open/py-recursion?lang=en).",
            "never": "Each call gets a number one smaller, and `f(0)` is the base case: the calls stop there. See *Recipe* in [A function that calls itself](/course/open/py-recursion?lang=en).",
        },
    )
    quick_recursion
    return (quick_recursion,)


@app.cell(hide_code=True)
def quick_check_dict():
    quick_dict = L.choice(
        "quick-dict",
        question="What does this program show?\n\n```python\nlevel = {\"desk\": 1, \"it\": 3}\nlevel[\"data\"] = 2\nprint(len(level), level[\"it\"])\n```",
        options={
            "two": "2 3",
            "error": "An error: \"data\" is not a key",
            "right": "3 3",
        },
        correct="right",
        why={
            "right": "The second line adds a third pair, so `len(level)` is 3, and the key `\"it\"` still has the value 3.",
            "two": "`level[\"data\"] = 2` adds a new pair when the key is not there yet: the dictionary now has three keys. See *New words and symbols* in [Dictionaries](/course/open/py-dicts?lang=en).",
            "error": "Giving a value to a key that is not there yet adds it; only looking up a missing key gives an error. See *New words and symbols* in [Dictionaries](/course/open/py-dicts?lang=en).",
        },
    )
    quick_dict
    return (quick_dict,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A made-up library, the one of the module privacy, keeps a member file: the names, e-mail addresses and phone numbers of its members. Keeping less of it, and deleting what is no longer needed, is about what the library does with the data itself. Security asks a different question: what would someone do to get at the data on purpose? Not "is the member file safe?", but "what is the cheapest way to read it, and for whom?". Thinking like an attacker means listing what can go wrong before someone else finds it, and then looking for the way in that costs least.

    **An everyday example.** A burglar does not pick the expensive lock on the front door when the window at the back is open. Making the front door stronger then changes nothing; closing the window does.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="In 1999, Bruce Schneier drew the ways to open a safe as a tree, with what each one would cost an attacker. Which way did the tree find cheapest?",
        options={
            "lock": "Pick the lock",
            "combo": "Learn the combination",
            "right": "Cut the safe open",
        },
        reveal="**Cut the safe open**, at $10K in Schneier's figure. Learning the combination costs $20K at the cheapest, by bribing the owner of the safe, and picking the lock $30K. *Attack trees* in *The idea* shows how the tree works that out.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Three things to protect

    Security protects three things above all. FIPS 199, a standard of NIST, the US standards institute, says that US law "defines three security objectives for information and information systems", and says what losing each one means:

    - **confidentiality**: "A loss of confidentiality is the unauthorized disclosure of information." Someone reads what they may not read.
    - **integrity**: "A loss of integrity is the unauthorized modification or destruction of information." Someone changes or deletes what they may not change.
    - **availability**: "A loss of availability is the disruption of access to or use of information or an information system." The people who may use the system cannot.

    Together they are often called the **CIA triad**, after their first letters. The three were not new in 2004: in 1975, Saltzer and Schroeder wrote that security specialists "have found it useful to place potential security violations in three categories": release, modification and "denial of use". They added a warning that still holds: "The biggest complication in a general-purpose remote-accessed computer system is that the 'intruder' in these definitions may be an otherwise legitimate user of the computer system." The attacker can be a member of staff.

    ### What can go wrong: STRIDE

    In 1999, Loren Kohnfelder and Praerit Garg wrote a note inside Microsoft with a list of six kinds of threat that a product team should look for during design. The first letters spell **STRIDE**: spoofing (pretending to be someone else), tampering (changing data), repudiation (denying you did something; their note says "Repudiability"), information disclosure, denial of service and elevation of privilege (getting rights you should not have). An article in MSDN Magazine in 2006, by Shawn Hernan, Scott Lambert, Tomasz Ostwald and Adam Shostack, "maps threats to the properties that guard against them": each kind of threat goes against one property. Three of those are the CIA triad; the other three are authentication (you are who you say you are), non-repudiation (an action is tied to whoever did it) and authorization (you only do what you are allowed to). The same article warns that one threat can fit two letters: "Don't get too hung up over the terminology." STRIDE helps you not to forget a kind of threat; it does not have to sort every threat perfectly.

    ### A threat model

    A **threat model** describes a system as an attacker would see it, so that you can defend it: what is worth attacking, who might attack, and how. The Threat Modeling Manifesto, written in 2020 by a working group of fifteen people, puts it in four questions: "What are we working on?" "What can go wrong?" "What are we going to do about it?" "Did we do a good enough job?" STRIDE helps with the second question, and so do attack trees.

    ### Attack trees

    Bruce Schneier described **attack trees** in 1999: "you represent attacks against a system in a tree structure, with the goal as the root node and different ways of achieving that goal as leaf nodes." Every node above the leaves is one of two kinds: "OR nodes are alternatives—the four ways to open a safe, for example. AND nodes represent different steps toward achieving the same goal." Give each leaf a value, and the tree works out the value of every node above it. For costs, the rule is: "OR nodes have the value of their cheapest child; AND nodes have the value of the sum of their children." The value at the root is then the cost of the cheapest attack. The same works for possible or impossible: an OR node is possible if one of its children is, an AND node only if all of them are.

    ### The cheapest way in

    An attacker does not need the hardest way in, only the cheapest one that works. Schneier's example: "breaking the RSA or IDEA encryption algorithms are not the most profitable attacks against PGP", a program for encrypted e-mail. Kohnfelder and Garg wrote the same in 1999: "the security of the entire system typically is no better than its weakest link." A tree is never finished, either. Schneier: "Of course there’s always the chance that you forgot about an attack, but you’ll get better with time."
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **CIA triad** | confidentiality, integrity and availability: the three things security protects |
        | **availability** | the people who may use a system or its data can use them, when they need to |
        | **STRIDE** | six kinds of threat to look for: spoofing, tampering, repudiation, information disclosure, denial of service and elevation of privilege |
        | **threat model** | a description of what in a system is worth attacking, who might attack it and how, so that you can defend it |
        | **attack tree** | a tree with an attacker's goal at the root and the ways to reach it below; OR nodes are alternatives, AND nodes steps that are all needed |
        """),
        L.watch_out("STRIDE in capitals is the list of threats; the stride of a convolution, in [Convolution in detail](/course/open/cnn-conv-formal?lang=en), has nothing to do with it. And in an attack tree, the cheapest step is not always the cheapest attack: under an AND node, its sibling steps have to be paid for too."),
        mo.Html(C.tree_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To think like an attacker about a system:

    1. Say what you protect, and which of the three it is about: confidentiality, integrity, availability.
    2. Go through STRIDE, and write down what could go wrong of each kind. That is `at_risk`.
    3. Draw an attack tree for one goal of the attacker: OR for alternatives, AND for steps that are all needed.
    4. Work out from the leaves up what the cheapest attack costs, and which steps it takes. That is `cost` and `cheapest`.
    5. Make that cheapest attack more expensive, and look again: the attacker moves to the next cheapest way.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.TREE` is the attack tree for reading the library's member file, as in the picture. Each node is a dict with a `"name"` and a `"kind"`: `"leaf"`, `"and"` or `"or"`. An `"and"` or `"or"` node has a list of `"children"`; a leaf has a `"cost"` in euros and `"possible"`, which says whether an attacker from outside, with little money, could take that step. The cell applies Schneier's rule for possible and impossible with a function that calls itself: a leaf answers directly, and a node counts how many of its children are possible.
    """)
    return


@app.cell
def example_possible():
    # possible or not, by Schneier's rule: an OR node needs one possible child, an AND node all of them
    def _possible(node):
        if node["kind"] == "leaf":
            return node["possible"]
        _count = 0
        for _child in node["children"]:
            if _possible(_child):
                _count = _count + 1
        if node["kind"] == "and":
            return _count == len(node["children"])
        return _count > 0

    for _way in C.TREE["children"]:
        print(_way["name"] + ":", _possible(_way))
    print("the goal:", _possible(C.TREE))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Logging in as a staff member is possible: it is an AND node, and both of its children, getting a password and getting the login code, have a possible step. Stealing the backup is not: getting into the IT room is easy, but breaking the encryption is not possible for this attacker, and an AND node needs both. Bribing is not possible without money. With two ways left that are possible, the goal is possible too. Schneier's rule says whether there is a way in; the exercises ask what the cheapest way costs.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_risk():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. at_risk: which properties are at risk

        `THREATS` is a list of eight threats against the library, each a dict with a `"what"` and its kind of STRIDE threat under `"stride"`, such as `"tampering"`. `PROPERTY` is a dictionary that gives, for each kind, the property it breaks: `PROPERTY["tampering"]` is `"integrity"`. `at_risk(threats)` gives back a dictionary with, for each property, how many of the threats break it. Inside, `counts` starts empty, and the loop goes over the threats. Replace the `...`. The check calls `at_risk(THREATS)`, then shorter lists.
        """),
        L.task("Count, for each property, how many threats break it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_risk():
    box_risk = L.code_box("at-risk", lines=10)
    box_risk
    return (box_risk,)


@app.cell(hide_code=True)
def check_risk(box_risk):
    result_risk = L.check("at-risk", box_risk.value)
    result_risk
    return (result_risk,)


@app.cell(hide_code=True)
def hints_risk():
    L.hints(
        "at-risk",
        [
            "Step 2 of the recipe. You counted with a dictionary in a loop in [Dictionaries](/course/open/py-dicts?lang=en): add 1 when the key is there, and start at 1 when it is not.",
            "For each threat, first look up in `PROPERTY` which property its kind breaks, and keep that in a name such as `prop`. Then count `prop` in `counts`.",
            '`prop = PROPERTY[___]`, then `if prop in counts:` with `counts[prop] = counts[prop] + 1` under it, and `else:` with `counts[prop] = 1`',
        ],
        why='`prop = PROPERTY[threat["stride"]]` gives the property; `if prop in counts:` adds 1 to its count, and `else:` starts it at 1. `at_risk(THREATS)` gives `{"authentication": 1, "integrity": 2, "non-repudiation": 1, "confidentiality": 2, "availability": 1, "authorization": 1}`.',
    )
    return


@app.cell(hide_code=True)
def text_after_risk():
    mo.md(r"""
    Five of the eight threats break one of the three properties of the CIA triad: two confidentiality, two integrity, one availability. The other three need what STRIDE adds: a stolen password breaks authentication, a waived fine that nobody will own up to breaks non-repudiation, and a desk account that can open the data team's files breaks authorization. A list like this is the start of a threat model. It does not say which threat matters most; for that you also need to know how bad each one would be, and what each attack would cost.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cost():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. cost: what the cheapest attack costs

        `cost(node)` gives back what the cheapest attack under `node` costs: for a leaf its `"cost"`, for an AND node the sum of what its children cost, and for an OR node what its cheapest child costs. A child can be a node with children of its own, so `cost` calls itself on each child. The first lines are there: a leaf gives back its own cost. Replace the `...`. The check calls `cost(TREE)`, then parts of the tree such as `LOGIN` (logging in as a staff member), Schneier's tree `SAFE` (in thousands of dollars), and small trees of its own.
        """),
        L.task("Give back the sum of the children for an AND node, and the cost of the cheapest child for an OR node."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cost():
    box_cost = L.code_box("cost", lines=13)
    box_cost
    return (box_cost,)


@app.cell(hide_code=True)
def check_cost(box_cost):
    result_cost = L.check("cost", box_cost.value)
    result_cost
    return (result_cost,)


@app.cell(hide_code=True)
def hints_cost():
    L.hints(
        "cost",
        [
            "Step 4 of the recipe, and the shape of the worked example: go over the children once, and decide by the kind of node afterwards.",
            'Before the loop, start a `total` at 0 and a `best` at the cost of the first child, `cost(node["children"][0])`. In the loop, add the cost of each child to `total`, and keep it in `best` when it is less. After the loop, an AND node gives back one of the two, an OR node the other.',
            '`if node["kind"] == "and":` with `return ___` under it, and after that `return best`',
        ],
        why='In the loop, `total = total + cost(child)` adds up the children, and `if cost(child) < best:` with `best = cost(child)` keeps the cheapest. After it, `if node["kind"] == "and":` gives back `total`, and otherwise `return best`. `cost(TREE)` gives `500`.',
    )
    return


@app.cell(hide_code=True)
def text_after_cost():
    mo.md(r"""
    Reading the member file costs €500 at the cheapest: logging in, which needs a password (€100 the cheapest way) and the login code (€400). Getting into the IT room costs only €50, the cheapest step in the tree, but stealing the backup also needs the encryption broken: €20,050. The SQL injection in the catalog search, the subject of the next lesson, costs €600, close behind. And Schneier's safe comes out at 10, the $10K of cutting it open.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cheapest():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. cheapest: the steps of the cheapest attack

        `cheapest(node)` gives back the names of the steps of the cheapest attack under `node`, as a list. A leaf gives back its own name in a list, and an AND node the steps of all its children: those lines are there. For an OR node, follow the child that costs least, and give back the steps under it; with two equally cheap children, the first of them. `cost` from Exercise 2 is ready to use, the course's version. Replace the `...`. The check calls `cheapest(TREE)`, then parts of the tree, `SAFE` and small trees.
        """),
        L.task("For an OR node, give back the steps of the cheapest attack under its cheapest child."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cheapest():
    box_cheapest = L.code_box("cheapest", lines=15)
    box_cheapest
    return (box_cheapest,)


@app.cell(hide_code=True)
def check_cheapest(box_cheapest):
    result_cheapest = L.check("cheapest", box_cheapest.value)
    result_cheapest
    return (result_cheapest,)


@app.cell(hide_code=True)
def hints_cheapest():
    L.hints(
        "cheapest",
        [
            "Step 4 of the recipe. An OR node is a choice: the attacker takes one child, the cheapest. Its steps are the steps of the cheapest attack under that child.",
            'Start with `best` as the first child, `node["children"][0]`. Go over the children, and keep a child in `best` when `cost` says it costs less than `best`. Then let `cheapest` find the steps under `best`.',
            '`if ___ < cost(best):` with `best = child` under it, and after the loop `return cheapest(best)`',
        ],
        why='`best = node["children"][0]`, then in the loop `if cost(child) < cost(best):` with `best = child`, and after the loop `return cheapest(best)`. `cheapest(TREE)` gives `["Guess the password", "Phish the code too"]`.',
    )
    return


@app.cell(hide_code=True)
def text_after_cheapest():
    mo.md(r"""
    The cheapest attack takes two steps: guess the password, then phish the login code too. Together they cost €500, the same as `cost(TREE)`. The tree now says where a defense helps most: on this path. Making the backup's encryption stronger would change nothing, because nobody would go that way anyway. For Schneier's safe, `cheapest(SAFE)` gives `["Cut Open Safe"]`.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Defenses

    Choose defenses for the library. Each one makes some steps more expensive, or takes a step away. The picture shows what each node costs now, with the steps of the cheapest attack filled in: with your `cost` and `cheapest` once Exercises 2 and 3 are right, and with the course's until then. Add them one at a time, in the order of the list, and watch where the attacker goes.
    """)
    return


@app.cell(hide_code=True)
def defenses_cell():
    defenses = mo.ui.multiselect(options=C.DEFENSE_LABELS[L.lang], label="Defenses")
    defenses
    return (defenses,)


@app.cell(hide_code=True)
def demo_defenses(box_cheapest, box_cost, defenses, result_cheapest, result_cost):
    _my_cost = C.mine(box_cost.value, result_cost.state == "correct", "cost")
    _my_cheapest = C.mine(box_cheapest.value, result_cheapest.state == "correct", "cheapest", {"cost": C.cost})
    _out = mo.Html(C.defense_picture(defenses.value, _my_cost, _my_cheapest, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Long passwords move the attacker to watching a password being typed at the desk: €550. Phishing training then makes the SQL injection the cheapest way in, at €600. Placeholders in the search take that step away, and logging in comes back, now with a stolen phone: €1,150. With screens turned away at the desk as well, the cheapest attack is bribing a staff member, for €2,000. Each defense helped, and each time the attacker simply took the next way in. That is why you look at the whole tree, and not only at the step that worried you first.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_integrity():
    mo.md(r"""
    ### Exercise 4. Which property?

    Someone changes the opening hours on the library's website, so that it says the library is closed on Saturdays. The website itself works as always.
    """)
    return


@app.cell(hide_code=True)
def choice_integrity_cell():
    choice_integrity = L.choice(
        "integrity",
        question="Which of the three does that break?",
        options={
            "avail": "Availability: on Saturdays people stay away from it",
            "right": "Integrity: the hours were changed without permission",
            "conf": "Confidentiality: everyone can now see the changed page",
        },
        correct="right",
        why={
            "right": "FIPS 199: \"A loss of integrity is the unauthorized modification or destruction of information.\" Someone changed the opening hours without the right to; in STRIDE that is tampering.",
            "avail": "The website still works, and everyone can still read the page: nothing is cut off. What went wrong is that the information was changed. See *Three things to protect* in *The idea*.",
            "conf": "Nothing secret was read: the opening hours are public. The information was changed. See *Three things to protect* in *The idea*.",
        },
    )
    choice_integrity
    return (choice_integrity,)


@app.cell(hide_code=True)
def text_exercise_and():
    mo.md(r"""
    ### Exercise 5. The cheapest step

    In the library's tree, getting into the IT room costs €50: the cheapest step of all.
    """)
    return


@app.cell(hide_code=True)
def choice_and_cell():
    choice_and = L.choice(
        "and-node",
        question="Why is stealing the backup still not the cheapest attack?",
        options={
            "right": "It is an AND node: the encryption has to be broken too",
            "outside": "The tree counts only what an outsider can do",
            "or": "Schneier's rule gives an OR node the cost of its most expensive child",
        },
        correct="right",
        why={
            "right": "\"AND nodes have the value of the sum of their children\": getting into the IT room (€50) plus breaking the encryption (€20,000) is €20,050.",
            "outside": "`cost` adds up costs; whether an attacker from outside can take a step is the worked example's question, not this one. See *Attack trees* in *The idea*.",
            "or": "In Schneier's rule, \"OR nodes have the value of their cheapest child\", not of the most expensive one. And stealing the backup is an AND node. See *Attack trees* in *The idea*.",
        },
    )
    choice_and
    return (choice_and,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [A function that calls itself](/course/open/py-recursion?lang=en), a function called itself on a smaller number or a shorter text; here it called itself on each child of a tree, and a base case, a leaf, stopped it. With [Dictionaries](/course/open/py-dicts?lang=en), you counted threats per property. The module privacy asked what an organization does with personal data; from here, the module security asks what an attacker could do with it. *Injection and secrets* takes the €600 step of this tree, the SQL injection, and shows how it works and how to close it.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_and, choice_integrity, result_cheapest, result_cost, result_risk):
    L.wrap_up(result_risk, result_cost, result_cheapest, choice_integrity, choice_and)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
