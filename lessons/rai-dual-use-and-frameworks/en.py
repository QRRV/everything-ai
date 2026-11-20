# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-dual-use-and-frameworks. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-dual-use-and-frameworks", "en")
    C = L.checks   # the frameworks, the dates of the AI Act and Google's counts, as the sources give them


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question on [Red teaming](/course/open/rai-red-teaming?lang=en) and one on [The OWASP lists](/course/open/rai-owasp-lists?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_disclosure():
    quick_disclosure = L.choice(
        "quick-disclosure",
        question="You notice that the help assistant of another bike shop gives away its staff code. Who should hear it first?",
        options={
            "right": "That shop, privately",
            "shops": "The other bike shops, so that they can check theirs",
            "online": "Everyone online, so that the shop fixes it quickly",
        },
        correct="right",
        why={
            "right": "That is responsible disclosure, which CERT and the Dutch NCSC call coordinated vulnerability disclosure: the maker first, the public once there is a fix. This lesson asks the same question of a whole list of working attacks.",
            "shops": "The other shops come later: first the shop whose assistant leaks hears it, privately, and gets time to fix it. See *Telling what you found* in [Red teaming](/course/open/rai-red-teaming?lang=en).",
            "online": "Online, anyone can use it on that shop before it has a fix. Responsible disclosure goes the other way: the maker first, the public once there is a fix. See *Telling what you found* in [Red teaming](/course/open/rai-red-teaming?lang=en).",
        },
    )
    quick_disclosure
    return (quick_disclosure,)


@app.cell(hide_code=True)
def quick_check_owasp():
    quick_owasp = L.choice(
        "quick-owasp",
        question="What is the OWASP Top 10 for LLM Applications for?",
        options={
            "test": "It is a test that an app passes before it goes online",
            "right": "It helps choose which risks of an app to look at first",
            "law": "It holds the rules that the EU sets for every app with an LLM",
        },
        correct="right",
        why={
            "right": "OWASP calls its Top 10 for the web \"a standard awareness document\": a list starts a review, it does not end it, and nobody has to follow it. This lesson meets more such documents, and one that is law.",
            "test": "No app passes or fails a Top 10: the list helps choose where to look first, and a review goes on from there. See *What an OWASP Top 10 is* in [The OWASP lists](/course/open/rai-owasp-lists?lang=en).",
            "law": "OWASP is not a government, and its lists bind no one. The EU does have a law on AI, the AI Act, which this lesson comes to. See *What an OWASP Top 10 is* in [The OWASP lists](/course/open/rai-owasp-lists?lang=en).",
        },
    )
    quick_owasp
    return (quick_owasp,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the previous lesson, Qwen3 wrote 30 test cases, and 23 of them got Spoke & Chain's staff code out of its help assistant. For the shop, that list is a gift: it shows what to fix. For someone who wants the code of a shop with the same assistant, the same list is the attack. Red teaming, the OWASP lists, a model that writes code: what helps a defender can help an attacker. This lesson is about that, and about the rules that people have written against it.

    Two questions run through it. What do attackers actually do with AI, according to the makers who see it happen? And which documents say what the maker of an AI system should do against it, and which of them are law?

    **An everyday example.** A ladder helps a painter reach a window, and helps a burglar reach it too. Nobody bans ladders. What matters is who uses one, for what, and what the house does to stay safe.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In November 2025, Anthropic reported a campaign in which attackers used its tool Claude Code against about thirty targets. What share of the campaign's work did the AI do, by Anthropic's account?",
        options={
            "small": "About a tenth",
            "middle": "About half",
            "right": "80 to 90%",
        },
        reveal="**80 to 90%.** Anthropic: the AI did \"80-90% of the campaign, with human intervention required only sporadically (perhaps 4-6 critical decision points per hacking campaign).\" It also writes that the attackers \"succeeded in a small number of cases\", and that the AI \"occasionally hallucinated credentials\", made up login details: a big share of the work, not a sure success. *What attackers do with AI* in *The idea* puts this report next to Google's.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### What dual use means

    **Dual use** is when the same thing serves a good purpose and a harmful one. Trade law uses the term too. The European Commission: "Dual-use items are goods, software and technology that can be used for both civilian and military applications." Their export is governed by Regulation (EU) 2021/821, which applies directly in every EU country.

    For AI, Executive Order 14110 of the US president, of October 2023, defined a "dual-use foundation model" as a model that "is trained on broad data; generally uses self-supervision; contains at least tens of billions of parameters; is applicable across a wide range of contexts; and that exhibits, or could be easily modified to exhibit, high levels of performance at tasks that pose a serious risk to security, national economic security, national public health or safety, or any combination of those matters". One of its examples of such a task: "enabling powerful offensive cyber operations through automated vulnerability discovery and exploitation against a wide range of potential targets of cyber attacks". And a safeguard did not change that: "Models meet this definition even if they are provided to end users with technical safeguards that attempt to prevent users from taking advantage of the relevant unsafe capabilities." The order was revoked in January 2025. Qwen3-0.6B, with 0.6 billion parameters, is far from tens of billions.

    The US AI Safety Institute, a government body that tests advanced AI models (a later lesson covers such institutes), wrote voluntary guidelines on the misuse of such models, published by NIST, the US standards institute; the version of January 2025 is a draft. The draft calls misuse risk "A risk that an AI model will be deliberately misused to cause harm." And in security, it writes, the line is hard to draw: "The dynamics of the offense-defense interplay in cyber mean that most capabilities will be, in some sense, dual use." "Reliably differentiating cyber misuse from legitimate use cases may be challenging." The test cases of the previous lesson are an example: the same messages tell the shop what to fix, and tell an attacker what works.

    ### What attackers do with AI

    The makers of models see part of it happen, in their own products, and some of them report it. In January 2025, the Google Threat Intelligence Group wrote about how attackers used Gemini, Google's model: "while AI can be a useful tool for threat actors, it is not yet the game-changer it is sometimes portrayed to be". They used it for "common tasks like troubleshooting, research, and content generation", and Google did "not see indications of them developing novel capabilities". "Rather than enabling disruptive change, generative AI allows threat actors to move faster and at higher volume." Jailbreaks were simple, and failed: the attackers used "publicly available jailbreak prompts in unsuccessful attempts to bypass Gemini's safety controls". Google found APT groups, short for advanced persistent threat, from more than 20 countries; it counted "Over 20 China-backed groups", "Over 10 Iran-backed groups", "Nine North Korea-backed groups" and "Three Russia-backed groups", and "Iranian APT actors were the heaviest users of Gemini".

    Anthropic, in August 2025, about AI that carries out tasks by itself: "Agentic AI has been weaponized." And "AI has lowered the barriers to sophisticated cybercrime." In one case of data extortion, where attackers steal data and demand money not to publish it: "The actor targeted at least 17 distinct organizations", with ransoms "that sometimes exceeded $500,000". In November 2025, Anthropic described the campaign of the guess, detected in mid-September 2025. The threat actor, "whom we assess with high confidence was a Chinese state-sponsored group", "manipulated our Claude Code tool into attempting infiltration into roughly thirty global targets and succeeded in a small number of cases." Anthropic: "We believe this is the first documented case of a large-scale cyberattack executed without substantial human intervention." The AI made mistakes too, and made things up: "It occasionally hallucinated credentials or claimed to have extracted secret information that was in fact publicly-available." And Anthropic corrected its own account of the speed afterwards, to "thousands of requests, often multiple per second".

    Each of these reports comes from a company about its own model. Google looked at Gemini, with data up to January 2025; Anthropic looked at Claude, and writes "although we only have visibility into Claude usage". None of them is an outside audit.

    ### The rules

    Several documents say what the maker of an AI system should do to keep it secure. In June 2023, Google introduced the Secure AI Framework (SAIF), "a conceptual framework for secure AI systems", with six core elements, from "Expand strong security foundations to the AI ecosystem" to "Contextualize AI system risks in surrounding business processes"; one of them asks for "regular red team exercises". Its site divides the SAIF map into "four component areas: Data, Infrastructure, Model, and Application", and adds: "It is not a reflection of Google's current technical implementations." SAIF 2.0, of October 2025, adds agents, AI systems that carry out tasks by themselves: "agents must have well-defined human controllers, their powers must be carefully limited, and their actions and planning must be observable."

    In November 2023, the UK's National Cyber Security Centre (NCSC), the US Cybersecurity and Infrastructure Security Agency (CISA) and their partners published *Guidelines for secure AI system development*. The document "recommends guidelines for providers of any systems that use artificial intelligence (AI)", in "four key areas within the AI system development life cycle: secure design, secure development, secure deployment, and secure operation and maintenance." In the words of the NCSC's news release: "Agencies from 18 countries, including the US, endorse new UK-developed guidelines on AI cyber security". No Dutch agency is on the list.

    In January 2025, the UK government published a voluntary *Code of Practice for the Cyber Security of AI*, with 13 principles, to help create a standard at ETSI, one of the three recognized European standards bodies. That standard is ETSI EN 304 223, adopted on 8 December 2025: "The present document defines baseline security requirements for AI models and systems." In ETSI's words, it "defines 13 principles and requirements across five phases: secure design, secure development, secure deployment, secure maintenance, and secure end of life." Principle 1 is "Raise awareness of AI security threats and risks", principle 13 "Ensure proper data and model disposal". On testing: "Developers shall ensure that all models, applications and systems that are released to System Operators and/or End-users have been tested as part of a security assessment process." A standard is not a law: the sources this lesson read do not say that any law requires it.

    One of these documents is law. The AI Act is the EU's law on AI, and its Article 15 is about high-risk AI systems; for the dates from which it applies, the Act splits them into the systems of its Annex III and those of its Annex I: "High-risk AI systems shall be designed and developed in such a way that they achieve an appropriate level of accuracy, robustness, and cybersecurity, and that they perform consistently in those respects throughout their lifecycle." And: "High-risk AI systems shall be resilient against attempts by unauthorised third parties to alter their use, outputs or performance by exploiting system vulnerabilities." The article names data poisoning, model poisoning, adversarial examples and attacks on confidentiality. The Act "shall apply from 2 August 2026", but after an amendment, the Digital Omnibus on AI, the rules for high-risk systems, with Article 15, apply from 2 December 2027 for the systems of Annex III, and from 2 August 2028 for those of Annex I.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.documents_picture(L.lang)),
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **dual use** | when the same thing serves a good purpose and a harmful one, such as a list of attacks that works |
        | APT group | advanced persistent threat: in Google's counts, a group of attackers backed by a state |
        | high-risk AI system | an AI system that the AI Act's rules for high risk apply to, such as Article 15 |
        | `C.NCSC_AREAS` | the four areas of the guidelines of the NCSC and CISA, 2023 |
        | `C.ETSI_PHASES` | the five phases of ETSI EN 304 223, 2025 |
        | `C.GROUPS`, `C.NUMBERS` | Google's counts per country, and the counts written as words with their numbers |
        """),
        L.watch_out("The dates of Article 15 are those of the AI Act as amended by the Digital Omnibus, in the text of 27 July 2026. Rules change: check the text that applies on the day you need it."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Before you share a finding, a tool or a model, ask whom else it helps: what helps you fix helps someone else attack.
    2. Read a report about attackers by who wrote it, about which model, and when.
    3. Use a framework as a checklist for the whole life of a system, from design to the day it is retired.
    4. Find out which rules are law for your system, and from which day; the rest are advice, however useful.
    5. Test before you release, as ETSI EN 304 223 asks, and keep testing after.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell compares the guidelines of 2023 with the standard of 2025. `C.NCSC_AREAS` holds the four areas of the guidelines and `C.ETSI_PHASES` the five phases of the standard, each as a list of texts. `area in etsi` asks whether the list `etsi` holds the text `area`.
    """)
    return


@app.cell
def example_same():
    ncsc = C.NCSC_AREAS    # the four areas of the guidelines of the NCSC and CISA, 2023
    etsi = C.ETSI_PHASES   # the five phases of ETSI EN 304 223, 2025
    print(len(ncsc), "areas,", len(etsi), "phases")

    same = []
    for area in ncsc:
        if area in etsi:
            same.append(area)
    print("the same name in both:", same)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Three names are the same in both. The standard has five phases, so two phases have no area of the same name. Do the guidelines leave them out, or name the same subject in other words? Comparing whole names is not enough to tell: Exercise 1 looks inside the names.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_uncovered():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. uncovered: what the standard adds

        Write `uncovered(phases, areas)`: the phases of the list `phases` that no area of the list `areas` covers, in the order of `phases`. An area covers a phase when the phase's name without `secure ` is part of the area's name: `maintenance` is part of `secure operation and maintenance`. Every name starts with `secure `, 7 characters, so `rest = phase[7:]` is the rest of the name. Between two texts, `in` asks whether the first is part of the second: `"cat" in "a cat sat"` is `True`. `covered` says whether an area holds `rest`, and the uncovered phases go in the list `found`. Replace the `...`. The check calls `uncovered(ETSI_PHASES, NCSC_AREAS)`, then a few lists of its own.
        """),
        L.task("Look through all the areas for each phase, and keep the phase when none holds rest."),
    ])
    return


@app.cell(hide_code=True)
def exercise_uncovered():
    box_uncovered = L.code_box("uncovered", lines=12)
    box_uncovered
    return (box_uncovered,)


@app.cell(hide_code=True)
def check_uncovered(box_uncovered):
    result_uncovered = L.check("uncovered", box_uncovered.value)
    result_uncovered
    return (result_uncovered,)


@app.cell(hide_code=True)
def hints_uncovered():
    L.hints(
        "uncovered",
        [
            "For each phase: start with `covered = False`, look at every area, and decide only after the last area whether the phase goes in `found`.",
            "Inside the loop over the areas, one `if` sets `covered = True`; after that loop, one `if` on `covered` adds the phase.",
            "`            if rest ___ area:`",
        ],
        why="`covered = False`, `for area in areas:` with `if rest in area:` and `covered = True` under it, then `if not covered:` with `found.append(phase)`. `uncovered(ETSI_PHASES, NCSC_AREAS)` gives `['secure end of life']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_uncovered():
    mo.md(r"""
    The guidelines of 2023 cover design, development, deployment and maintenance; the standard of 2025 adds the end: what happens to the data and the model when a system is retired. A model that leaves the building on an old disk can still give away what it learned. Comparing two documents by the words inside their names is quick; it is not reading them, and a phase with another name could still be covered under another heading.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_article():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. article_15: when the law applies

        Write `article_15(kind, day)`: whether Article 15 of the AI Act applies, on the day `day`, to a system of the kind `kind`, as `True` or `False`. `kind` is `"Annex III"` or `"Annex I"` for a high-risk AI system of that annex, and any other text for a system that is not high-risk. `day` is a date written as year-month-day, such as `"2027-12-02"`. Two texts compare character by character, so `"2027-12-02" < "2028-01-15"` is `True`: dates written this way compare like the days themselves. A rule applies from its first day on, and that day counts. Replace the `...`. The check calls `article_15("Annex III", "2027-12-02")`, then more kinds and days.
        """),
        L.task("Give each annex its own first day, and every other kind False."),
    ])
    return


@app.cell(hide_code=True)
def exercise_article():
    box_article = L.code_box("article-15", lines=7)
    box_article
    return (box_article,)


@app.cell(hide_code=True)
def check_article(box_article):
    result_article = L.check("article-15", box_article.value)
    result_article
    return (result_article,)


@app.cell(hide_code=True)
def hints_article():
    L.hints(
        "article-15",
        [
            "*The rules* in *The idea* gives two days: one for the systems of Annex III, one for those of Annex I. A system that is not high-risk has none.",
            "One `if` for each annex, each returning a comparison of `day` with the first day of that annex; after them, return `False`.",
            "`        return day ___ \"2027-12-02\"`",
        ],
        why="`if kind == \"Annex III\":` with `return day >= \"2027-12-02\"`, `if kind == \"Annex I\":` with `return day >= \"2028-08-02\"`, then `return False`. `article_15(\"Annex III\", \"2027-12-02\")` gives `True`.",
    )
    return


@app.cell(hide_code=True)
def text_after_article():
    mo.md(r"""
    On 2 August 2026 the AI Act applies, but Article 15 does not yet, to any system: a date in a headline is not the date of every rule. And a help assistant that is not a high-risk AI system never falls under Article 15, whatever the day. For such a system, SAIF, the guidelines and the standard are advice: worth following, but none of them is a law.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_groups():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. at_least: reading a count

        `C.GROUPS` holds Google's counts of the groups backed by a state that used Gemini, `{"China": "Over 20", "Iran": "Over 10", "North Korea": "Nine", "Russia": "Three"}`, and `C.NUMBERS` the counts written as words, with their numbers, `{"Nine": 9, "Three": 3}`. Write `at_least(groups, numbers)`: the fewest groups these counts allow, as a whole number. More than 20 groups is at least 21. `count[:5]` is the first 5 characters of the text `count`, and `int("20")` gives the number `20`. `for country, count in groups.items():` takes each country with its count, and `total` is the sum so far. Replace the `...`. The check calls `at_least(GROUPS, NUMBERS)`, then counts of its own.
        """),
        L.task("Add each country's fewest groups to total: Over n is n + 1, a word is its number."),
    ])
    return


@app.cell(hide_code=True)
def exercise_groups():
    box_groups = L.code_box("at-least", lines=9)
    box_groups
    return (box_groups,)


@app.cell(hide_code=True)
def check_groups(box_groups):
    result_groups = L.check("at-least", box_groups.value)
    result_groups
    return (result_groups,)


@app.cell(hide_code=True)
def hints_groups():
    L.hints(
        "at-least",
        [
            "Two kinds of count: one starts with `Over ` and a number, the other is a word that `numbers` knows. Each needs its own sum.",
            "An `if` on `count[:5]`; under it, add to `total` what the count after `Over` allows at the fewest; under `else:`, add the number from `numbers`.",
            "`            total = total + ___ + 1`",
        ],
        why="`if count[:5] == \"Over \":` with `total = total + int(count[5:]) + 1`, and under `else:` `total = total + numbers[count]`. `at_least(GROUPS, NUMBERS)` gives `44`.",
    )
    return


@app.cell(hide_code=True)
def text_after_groups():
    mo.md(r"""
    At least 44 groups backed by four states used Gemini, and Google's report speaks of groups from more than 20 countries. Those are the groups that one company saw in its own model; what attackers do with other models, or with models that run on their own computers, is not in this count.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_dual():
    mo.md(r"""
    ### Exercise 4. A list that works

    Spoke & Chain has fixed its assistant. It wants to post the 23 test cases that got the code out, so that other bike shops can test their own assistants.
    """)
    return


@app.cell(hide_code=True)
def choice_dual_cell():
    choice_dual = L.choice(
        "dual-use",
        question="Why is that list dual use?",
        options={
            "right": "It helps shops test, and helps attackers attack untested ones",
            "harmless": "It is not: the list holds no code, only questions",
            "law": "Because EU export law counts every red team test as a dual-use item",
        },
        correct="right",
        why={
            "right": "The messages that showed Spoke & Chain what to fix work on any assistant with the same weak spot, also for someone who wants its code. In NIST's words, \"most capabilities will be, in some sense, dual use\". A careful way: tell the makers of such assistants first, as with any weak spot, and post the list once they could fix it.",
            "harmless": "The messages are the attack: 23 of 30 got the code out of an assistant that was told to keep it secret. A list of working attacks without a code in it still works. See *What dual use means* in *The idea*.",
            "law": "The EU's export rules are about goods, software and technology for civilian and military use, and nothing this lesson read names red team tests. Dual use here is the plain meaning: the same thing helps both sides. See *What dual use means* in *The idea*.",
        },
    )
    choice_dual
    return (choice_dual,)


@app.cell(hide_code=True)
def text_exercise_reports():
    mo.md(r"""
    ### Exercise 5. Two reports

    Google wrote in January 2025 that AI is "not yet the game-changer it is sometimes portrayed to be" for attackers. Anthropic wrote in November 2025 that AI did 80 to 90% of the work of one campaign.
    """)
    return


@app.cell(hide_code=True)
def choice_reports_cell():
    choice_reports = L.choice(
        "reports",
        question="Do the two reports contradict each other?",
        options={
            "error": "Yes: one of the two companies must have it wrong",
            "right": "Not outright: other times, other models, each seen by its maker",
            "both": "No: both show that AI now breaks into most of its targets by itself",
        },
        correct="right",
        why={
            "right": "Google looked at Gemini, with data up to January 2025; Anthropic at Claude Code, in September 2025; and each sees only its own model, as Anthropic writes: \"although we only have visibility into Claude usage\". Read a report by who wrote it, about which model, and when.",
            "error": "Both can be right: they describe other months and other models, each from inside one company. See *What attackers do with AI* in *The idea*.",
            "both": "Anthropic's campaign aimed at roughly thirty targets and \"succeeded in a small number of cases\", and its AI \"occasionally hallucinated credentials\". Google saw attackers move faster, not new capabilities. See *What attackers do with AI* in *The idea*.",
        },
    )
    choice_reports
    return (choice_reports,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Red teaming](/course/open/rai-red-teaming?lang=en), you found the weak spots of your own assistant; the same test cases would serve an attacker, and that is dual use. [The OWASP lists](/course/open/rai-owasp-lists?lang=en) say which risks to look at first; the documents here say what to do over the whole life of a system, and the AI Act makes part of it law for high-risk systems. The next lesson, *Where did this picture come from?*, turns to what AI makes: how a maker can sign a picture, so that anyone can check where it came from.

    **An everyday example.** Kitchen knives are sold everywhere, and they can hurt someone. A cook learns to use them safely, a restaurant follows the hygiene rules that are law, and the guidebook of a chefs' association is advice. AI security has all three: skills, advice and, for some systems, law.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_dual, choice_reports, result_article, result_groups, result_uncovered):
    L.wrap_up(result_uncovered, result_article, result_groups, choice_dual, choice_reports)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
