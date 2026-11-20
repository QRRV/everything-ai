# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-kinds-of-ai-risk. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow category

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-kinds-of-ai-risk", "en")
    C = L.checks   # nine made-up cases of AI risk, and the seven domains of the MIT AI Risk Repository


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_allocative():
    quick_allocative = L.choice(
        "quick-allocative",
        question="A model decides who gets an extra check at the border, and people of one group get far more checks than others. Which kind of harm is that?",
        options={
            "representational": "Representational: it shows that group in a bad light",
            "right": "Allocative: it is about what the system hands out to people",
            "none": "No harm yet, since a check is not a final decision about anyone",
        },
        correct="right",
        why={
            "right": "An extra check is something the system hands out, and one group gets more of it: allocative harm. This lesson sorts risks in another way, by where they come from.",
            "representational": "Representational harm is about how a system shows or describes groups. Here the system hands out checks. See [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).",
            "none": "Extra checks cost time and trust, and they are handed out unequally: that is allocative harm. See [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).",
        },
    )
    quick_allocative
    return (quick_allocative,)


@app.cell(hide_code=True)
def quick_check_representational():
    quick_representational = L.choice(
        "quick-representational",
        question="How do you find representational harm in what a model writes?",
        options={
            "right": "Collect many outputs, and count who is shown how",
            "yes": "Compare the share of each group that gets a yes",
            "ask": "Ask the model whether its own outputs contain stereotypes",
        },
        correct="right",
        why={
            "right": "Representational harm needs no decision about anyone, so you have to look for it in many outputs. This lesson asks what other kinds of risk there are.",
            "yes": "That measures allocative harm: what a system hands out. Representational harm is in how groups are shown. See [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).",
            "ask": "A model that shows a stereotype need not notice it. You collect many outputs and count who is shown how. See [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en).",
        },
    )
    quick_representational
    return (quick_representational,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A chatbot that promises a refund the shop does not offer, a cloned voice of a manager used to have staff transfer money, translators who get fewer jobs: all three are risks of AI, and each asks for a different answer. The first is fixed in the system. The second needs someone to stop the person who misuses it. The third is fixed by neither, since nobody meant harm and every system did what it was meant to do.

    This lesson starts the module on AI safety. It sorts risks in two ways, one from 2019 and one from the International AI Safety Report, writes the sorting as a function, and shows why a list of kinds helps: a list written from memory leaves things out.

    **An everyday example.** A packing list sorted into clothes, papers and medicine shows at a glance that there is no passport on it. A list written from memory does not.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Researchers at MIT collected 74 documents that each list risks of AI, and sorted the risks they name into 24 subdomains. How many of the 24 does one document cover, on average?",
        options={
            "few": "About 3",
            "right": "About 8",
            "most": "About 20",
        },
        reveal="**About 8.** Slattery et al. (2026): \"Frameworks typically addressed only a subset of risks—averaging 8 of our identified 24 subdomains, with coverage ranging from 1 to 20 subdomains.\" *Why sort at all?* in *The idea* says what follows from that.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### What AI safety is about

    The International AI Safety Report, which this lesson comes back to, defines in its edition of 2026 the safety of an AI system as "The property of an AI system being unlikely to cause harm, whether through malicious misuse or system malfunctions." **AI safety** is the field that works on that: keeping AI systems from doing harm, whether people misuse them or they fail. That covers more than one fear: the report of 2025 counts unreliable answers and bias among its risks, next to losing control; the edition of 2026 has a smaller scope, and no longer addresses issues such as bias, the environment, privacy and copyright. In 2016, Amodei et al. used the name more narrowly, for "mitigating accident risk (often referred to as 'AI safety' in public discussions)".

    A related word: **alignment** is, in the words of the edition of 2026, "The propensity of an AI model or system to use its capabilities in line with human intentions, values, or norms." Put simply, an aligned model tends to do what people intend. Later lessons of this module show how that goes wrong.

    ### Accidents, misuse and structure

    In 2019, Remco Zwetsloot and Allan Dafoe wrote: "Dividing AI risks into misuse risks and accident risks has become a prevailing approach in the field." **Misuse** is someone using AI to do harm: "Misuse risks entail the possibility that people use AI in an unethical manner, with the clearest cases being those involving malicious motivation." An **accident** is harm from a system that does not do what was intended: "Accident risks, in contrast, involve harms arising from AI systems behaving in unintended ways. A prototypical example might be a self-driving car collision arising from the AI misunderstanding its environment."

    They add a third perspective. Misuse and accidents "tend to focus only on the last step in a causal chain leading up to a harm: that is, the person who misused the technology, or the system that behaved in unintended ways." But technology also changes "the environment and incentives (the 'structure' of a situation)". A **structural risk** comes from that structure, not from one person or one failing system: "in many situations the level of risk would basically be left unchanged even after a change in one agent’s behavior. As with an avalanche, it may be more useful to ask what caused the slope to become so steep, rather than what specific event set it off." Their examples for AI are nuclear deterrence and cyber operations, and they point to what others have cited: monopolistic markets, labor displacement and privacy erosion. Railways before the First World War, whose role historians still debate, illustrate the point, they write: technologies "even when they are not deliberately misused and behave just as intended, could have potentially far-reaching negative effects." The name structural risk is a short form: they speak of a structural perspective, and write that "structural risks are also collective risks".

    ### The International AI Safety Report

    The **International AI Safety Report** reviews what science knows about the capabilities and risks of general-purpose AI, models such as chat models that can do many kinds of tasks. Yoshua Bengio chairs it. 96 AI experts contributed to the first full report, of January 2025, including an advisory panel nominated by 30 countries, the OECD, the EU and the UN. That report says of itself: "It does not recommend specific policies." The edition of February 2026 was written with guidance from over 100 experts. Both sort risks into three categories; in 2025: "This report classifies general-purpose AI risks into three categories: malicious use risks; risks from malfunctions; and systemic risks."

    - Malicious use, in the glossary of 2025: "Employing AI to intentionally cause harm."
    - A malfunction, in the same glossary: "The failure of a general-purpose AI system to operate as intended by its developer or user, resulting in incorrect or harmful outputs or operational disruptions."
    - Systemic risks, in the edition of 2026: "Risks that arise from how AI development and deployment changes human behaviour, organisational practices, or societal structures, rather than directly from AI capabilities."

    They look like misuse, accidents and structure, but they are not the same. For Zwetsloot and Dafoe, misuse is any unethical use, with malicious motivation as its clearest case; the report's malicious use is harm on purpose. And structure is for them less a third box than a perspective that complements the other two. The report of 2025 puts bias under malfunctions, next to reliability issues and loss of control. And the edition of 2026 warns: "These categories are not exhaustive or mutually exclusive – risks may cut across multiple categories". A risk in its list "does not necessarily imply a risk is likely, severe, or requires policy action."
    """)
    return


@app.cell(hide_code=True)
def report_cell():
    mo.Html(C.report_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Loss of control

    **Loss of control** is, in the glossary of 2025, "A scenario in which one or more general-purpose AI systems come to operate outside of anyone’s control, with no clear path to regaining control." It need not be dramatic: "In ‘passive’ loss of control scenarios, important decisions are delegated to AI systems, but the systems’ decisions are too opaque, complex, or fast to allow for or incentivise meaningful oversight." How likely is it? The same report: "expert opinion on the likelihood of loss of control within the next several years varies greatly: some consider it implausible, some consider it likely to occur, and some see it as a modest-likelihood risk that warrants attention due to its high potential severity." One point is broadly shared: "There is broad consensus that current general-purpose AI lacks the capabilities to pose this risk."

    ### Why sort at all?

    The MIT AI Risk Repository, by Slattery et al., collects what others wrote: "a living database of 1,725 risks extracted from 74 existing taxonomies and frameworks", in its version of March 2026. A taxonomy is a set of kinds to sort things into. The repository sorts the risks twice. Its Causal Taxonomy asks "which entity causes them (human or AI), whether intentional, and when they occur (before or after deployment)": 42% of the risks are put down to AI systems and 38% to human decisions, and 35% are intentional and 35% unintentional. Its Domain Taxonomy has seven domains and 24 subdomains, from discrimination and toxicity to the safety of AI systems themselves. And no document covers all 24 subdomains: one covers 8 of them on average, from 1 to 20. That is what a taxonomy is for: a list of kinds shows what a list from memory leaves out.
    """)
    return


@app.cell(hide_code=True)
def mit_cell():
    mo.Html(C.mit_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **AI safety** | the field that works on keeping AI systems from doing harm, whether people misuse them or they fail |
        | **alignment** | how much an AI system tends to use what it can do in line with what people intend |
        | **misuse** | someone using AI to do harm, most clearly on purpose |
        | **accident** | harm from an AI system that does not do what was intended |
        | **structural risk** | risk from how AI changes the environment and incentives, not from one person or one failing system |
        | **International AI Safety Report** | a review, chaired by Yoshua Bengio, of what science knows about the capabilities and risks of general-purpose AI |
        | malicious use, malfunction, systemic risk | the report's three categories |
        | **loss of control** | AI systems that operate outside of anyone's control, with no clear path to regaining it |
        | `C.RISKS` | nine made-up cases: dicts with `"text"`, `"harm_meant"` and `"worked"` |
        | `C.DOMAINS`, `C.CHECKLIST` | the seven domains of the MIT repository, and the three a made-up team wrote down |
        | `elif` | else if: Python checks this condition only when the `if` before it was false |
        | `x not in a_list` | `True` when the list does not hold `x` |
        """),
        L.watch_out("Sorting a risk says what kind it is, not how likely or how bad it is. The report itself says that a risk in its list is not necessarily likely or severe."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        ### Recipe

        1. Ask whether someone meant to cause harm with AI. If so, it is malicious use, closest to what Zwetsloot and Dafoe call misuse.
        2. If not, ask whether the system did what its developer or user intended. If not, it is a malfunction, closest to an accident.
        3. If it did, and harm still comes from how AI is used across society, it is a systemic risk, closest to structure.
        4. Keep in mind that one risk can belong to more than one category.
        5. Check a list of risks against a taxonomy, to find what it leaves out.
        """),
        mo.Html(C.tree_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell takes the second of the course's cases and prints its text and its two answers. `C.RISKS[1]` is a dict; `case["worked"]` looks up the value under the key `"worked"`.
    """)
    return


@app.cell
def example_case():
    case = C.RISKS[1]   # the second case
    print(case["text"])
    print("harm meant:", case["harm_meant"])
    print("worked as intended:", case["worked"])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Nobody meant harm, and the chatbot did not do what the shop intended: by the recipe, a malfunction. The report would call it a reliability issue, Zwetsloot and Dafoe an accident. The fix lies in the system: a chatbot that only states the shop's real policy.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_category():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. iasr_category: three categories

        `C.RISKS` holds nine made-up cases, each a dict with the keys `"text"`, `"harm_meant"` (did someone mean to cause harm?) and `"worked"` (did the system do what its developer or user intended?). Write `iasr_category(case)`: the category of the dict `case` among the report's three, as one of the strings `"malicious use"`, `"malfunction"` or `"systemic"`. Ask the questions of the recipe in their order, with `if` and then `elif`, which Python only checks when the `if` before it was false. Replace the `...`. The check calls `iasr_category(RISKS[0])`, then the other cases and one of its own.
        """),
        L.task("Ask whether someone meant harm, and else whether the system did what was intended; give back the category."),
    ])
    return


@app.cell(hide_code=True)
def exercise_category():
    box_category = L.code_box("iasr-category", lines=6)
    box_category
    return (box_category,)


@app.cell(hide_code=True)
def check_category(box_category):
    result_category = L.check("iasr-category", box_category.value)
    result_category
    return (result_category,)


@app.cell(hide_code=True)
def hints_category():
    L.hints(
        "iasr-category",
        [
            "*Recipe* gives the two questions and their order.",
            "The first question returns the first category when someone meant harm; the second question then asks whether the system did what was intended; the last line returns the category that is left.",
            "`    elif ___:`",
        ],
        why="`if case[\"harm_meant\"]:` with `return \"malicious use\"` under it, then `elif case[\"worked\"]:` with `return \"systemic\"`, and last `return \"malfunction\"`. `iasr_category(RISKS[0])` gives `\"malicious use\"`.",
    )
    return


@app.cell(hide_code=True)
def text_after_category():
    mo.md(r"""
    Three of each. The fake reviews, the cloned voice and the posts before an election are malicious use, although each model did just what it was asked: misuse needs no broken system. In the report of 2025, the model that ranks women lower is a malfunction, and the electricity use of data centers a systemic risk, although no single system does anything wrong.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: follow a case

    Pick a case, and see which way it takes through the two questions: its category in the report, and the closest of the three of Zwetsloot and Dafoe.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_case = mo.ui.dropdown(options=C.case_options(L.lang), value=list(C.case_options(L.lang))[0], label="A case")
    demo_case
    return (demo_case,)


@app.cell(hide_code=True)
def demo_look(demo_case):
    mo.Html(C.demo_picture(demo_case.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_missing():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. missing: what a list leaves out

        The seven domains of the MIT repository are in `C.DOMAINS`. A made-up team that builds a chatbot for a bike shop wrote down risks under three of them, in `C.CHECKLIST`. Write `missing(covered, domains)`: the domains of the list `domains`, in their order, that are not in the list `covered`. Collect them in the list `left`, with a `for` loop that takes each domain as `domain`; `x not in a_list` is `True` when the list does not hold `x`. Replace the `...`. The check calls `missing(CHECKLIST, DOMAINS)`, and then lists of its own.
        """),
        L.task("Go over the domains, and keep each one that the list covered does not hold."),
    ])
    return


@app.cell(hide_code=True)
def exercise_missing():
    box_missing = L.code_box("missing", lines=6)
    box_missing
    return (box_missing,)


@app.cell(hide_code=True)
def check_missing(box_missing):
    result_missing = L.check("missing", box_missing.value)
    result_missing
    return (result_missing,)


@app.cell(hide_code=True)
def hints_missing():
    L.hints(
        "missing",
        [
            "*Why sort at all?* in *The idea* says what a taxonomy is for: here you compare a list with it.",
            "`left` starts as an empty list; in the loop, an `if` decides whether the domain is appended to it; after the loop, the function returns `left`.",
            "`        if ___:`",
        ],
        why="`left = []`, `for domain in domains:`, then `if domain not in covered:` with `left.append(domain)` under it, and `return left`. `missing(CHECKLIST, DOMAINS)` gives the four other domains, from `\"Malicious actors & misuse\"` to `\"AI system safety, failures & limitations\"`.",
    )
    return


@app.cell(hide_code=True)
def text_after_missing():
    mo.md(r"""
    The team thought of what its chatbot shows people and what it knows about them, and left out misuse by others, how people deal with the chatbot, its effect on jobs and the environment, and failures of the system itself. Documents leave things out too: across 74 of them, the MIT authors found that one covers 8 of the 24 subdomains on average.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_uber():
    mo.md(r"""
    ### Exercise 3. A brake that was switched off

    In 2018, a self-driving test car of Uber struck and killed a pedestrian. Commentators first blamed the car's vision system. But, Zwetsloot and Dafoe write, "the vehicle in fact detected the victim early enough for the emergency braking system to prevent a crash." The problem was that "the emergency brake had purposely been turned off by engineers who were afraid that an overly sensitive braking system would make their vehicle look bad relative to competitors."
    """)
    return


@app.cell(hide_code=True)
def choice_uber_cell():
    choice_uber = L.choice(
        "uber",
        question="What does the structural perspective add to this crash?",
        options={
            "accident": "Nothing: the car did not do what was intended, so it stays an accident",
            "right": "Competition with other firms made switching off the brake look sensible",
            "misuse": "It turns the crash into misuse, since the brake was switched off on purpose",
        },
        correct="right",
        why={
            "right": "Nobody meant harm, but competition pushed the engineers to switch the brake off: the slope was made steep before anything set it off. Zwetsloot and Dafoe: \"Safety, in short, is not just a technical challenge but also a political one.\"",
            "accident": "That is the last step of the chain. The structural perspective asks what came before it: engineers who feared their car would look bad next to competitors. See *Accidents, misuse and structure* in *The idea*.",
            "misuse": "The brake was switched off on purpose, but nobody meant to cause harm; the clearest cases of misuse involve \"malicious motivation\". See *Accidents, misuse and structure* in *The idea*.",
        },
    )
    choice_uber
    return (choice_uber,)


@app.cell(hide_code=True)
def text_exercise_overlap():
    mo.md(r"""
    ### Exercise 4. One flaw, many places

    Many apps build on the models of a few providers. Suppose one of those models has a flaw, and every app that builds on it fails in the same way at the same time.
    """)
    return


@app.cell(hide_code=True)
def choice_overlap_cell():
    choice_overlap = L.choice(
        "overlap",
        question="Where does that belong among the report's three categories?",
        options={
            "malfunction": "Only under malfunctions: the model does not do what was intended",
            "right": "Under two: a malfunction that becomes a systemic risk",
            "misuse": "Under malicious use, as soon as someone finds the flaw and exploits it",
        },
        correct="right",
        why={
            "right": "The flaw is a malfunction, and that so many apps fail at once comes from market concentration: the report of 2025 lists \"single points of failure\" among its systemic risks, and its edition of 2026 warns that its categories \"are not exhaustive or mutually exclusive\".",
            "malfunction": "That is true of the flaw, but not of why so many apps fail at once: the report of 2025 lists \"single points of failure\" among its systemic risks. See *The International AI Safety Report* in *The idea*.",
            "misuse": "In this case nobody exploits the flaw, and nobody means harm. It is a malfunction, and because so many apps depend on one model, a systemic risk as well.",
        },
    )
    choice_overlap
    return (choice_overlap,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Two kinds of harm](/course/open/rai-two-kinds-of-harm?lang=en) sorted harm by what it does to people: what a system hands out, and how it shows people. This lesson sorted risks by where they come from: someone who means harm, a system that fails, or a structure in which nobody does anything wrong. The International AI Safety Report calls them malicious use, malfunctions and systemic risks; Zwetsloot and Dafoe speak of misuse, accidents and structure. A taxonomy such as MIT's shows what a list from memory leaves out. The next lesson, *Concrete problems in AI safety*, looks at five problems of systems that learn, named in 2016.

    **An everyday example.** At a checkup, a doctor goes down the same list of questions every time: not because each problem is likely, but so that none is skipped.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_overlap, choice_uber, result_category, result_missing):
    L.wrap_up(result_category, result_missing, choice_uber, choice_overlap)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
