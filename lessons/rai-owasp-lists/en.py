# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-owasp-lists. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow improper output handling, overreliance

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-owasp-lists", "en")
    C = L.checks   # OWASP's lists for LLM applications and for agents, and the review of the library's assistant


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions on [Attacks on machine learning](/course/open/rai-ml-threat-models?lang=en) and [Prompts that check themselves](/course/open/start-prompt-chains?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_goal():
    quick_goal = L.choice(
        "quick-goal",
        question="Someone sends a model so many questions that other users no longer get their answers in time. Which of NIST's goals does that attacker have?",
        options={
            "right": "Availability breakdown: others lose access",
            "integ": "Integrity violation: the answers become wrong",
            "priv": "Privacy compromise: the answers give away what it learned",
        },
        correct="right",
        why={
            "right": "NIST: the attacker disrupts \"the ability of other users or processes to obtain timely and reliable access to its services.\" On OWASP's list for apps with an LLM, that risk falls under Unbounded Consumption.",
            "integ": "The answers stay right; they only come too late, or not at all. See *What attackers want from a model* in [Attacks on machine learning](/course/open/rai-ml-threat-models?lang=en).",
            "priv": "Nothing leaks: the attacker learns nothing from the answers. What goes wrong is that others can no longer use the model. See *What attackers want from a model* in [Attacks on machine learning](/course/open/rai-ml-threat-models?lang=en).",
        },
    )
    quick_goal
    return (quick_goal,)


@app.cell(hide_code=True)
def quick_check_citation():
    quick_citation = L.choice(
        "quick-citation",
        question="An assistant answers from a text you give it, and names the sentence it used. What does that let you do?",
        options={
            "right": "Check the answer against that sentence",
            "trust": "Trust the answer without checking it",
            "learn": "Make the model remember the text for next time",
        },
        correct="right",
        why={
            "right": "That is grounding with a citation: the named sentence is there to check the answer against. The library's assistant in this lesson does the same with the catalog record it used.",
            "trust": "A model can still get the answer wrong, also with a text. The named sentence is there so that you can check. See *The idea* in [Prompts that check themselves](/course/open/start-prompt-chains?lang=en).",
            "learn": "The text is only in the prompt; the model does not learn from it. What the citation gives you is a sentence to check the answer against. See *The idea* in [Prompts that check themselves](/course/open/start-prompt-chains?lang=en).",
        },
    )
    quick_citation
    return (quick_citation,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    So far in this module, attacks went for a website, its database and a model that reads pictures. Many apps now put a large language model, an LLM, in the middle: the kind of model behind a chat assistant, which reads text and writes text. The made-up library is about to get one, an assistant that answers members' questions about the catalog and renews their loans. Before it goes live, someone has to look for what can go wrong. This lesson is about the lists that say where to start: OWASP's Top 10 for LLM applications, and its list for agents. You will look up entries across the editions of the list, see what moved, and write the findings of a first review of the assistant.

    **An everyday example.** At a yearly car inspection, the inspector works through a fixed list: brakes, lights, tires. A car that passes can still break down the next day, and the list is updated now and then. Nobody takes the inspection for the whole of car safety, and nobody skips it either.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="For its 2026 list, OWASP let data on 6,639 real incidents count for the first time. Practitioners rank prompt injection the number one risk. Where would it be if the list were ranked by the incidents alone?",
        options={
            "first": "First, as in the vote",
            "five": "Lower, but still in the top five",
            "right": "Outside the top 10 altogether",
        },
        reveal="**Outside the top 10.** OWASP: \"Rank the categories by the raw incident record instead, and it falls out of the top 10 entirely.\" Yet prompt injection is first again in 2026: \"It stays at number one.\" *The idea* says why.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### What an OWASP Top 10 is

    The OWASP Foundation, which [Injection and secrets](/course/open/rai-classic-vulnerabilities?lang=en) quoted on SQL injection, keeps a list of the biggest risks for websites and web apps: "The OWASP Top 10 is a standard awareness document for developers and web application security. It represents a broad consensus about the most critical security risks to web applications." Apps with an LLM have a list of their own, the **OWASP Top 10 for LLM Applications**. In OWASP's words: "The list guides developers, security professionals, and organizations as they prioritize their efforts to identify and mitigate critical generative AI application security risks." So it helps to choose what to look at first. It is not a test that an app passes: an app can have none of the ten and still have a hole that is on no list.

    ### Three editions

    The list has had three editions, and each says how it was made.

    - **2023** (version 1.1, October 2023): nearly 500 experts took part, with "team members writing up 43 distinct threats", which "multiple rounds of voting" cut to ten.
    - **2025** (published in November 2024): "The process involved brainstorming sessions, voting, and real-world feedback from professionals in the thick of LLM application security".
    - **2026** (announced in September 2026), for the first time with data: "Every version of this list before now was built on judgment." OWASP collected 7,714 real incidents, and its classifiers sorted the 6,639 that carried enough detail. "The community vote carries three-quarters of the weight. The incident data covers the remaining quarter."

    Each entry has an id, LLM01 to LLM10, in the order of the list. The same id can be a different entry in another edition: LLM03 is Training Data Poisoning in 2023, Supply Chain in 2025 and Excessive Agency in 2026. So a finding names its edition after the id, as OWASP itself writes LLM01:2025. Names change too: what was System Prompt Leakage in 2025 is Hidden Context Exposure in 2026, and the Overreliance of 2023 is, in 2025, a related issue under Misinformation.

    ### The ten of 2026

    In plain words, after OWASP's descriptions in its lists of 2025 and 2026:

    | id | entry | what goes wrong |
    |---|---|---|
    | LLM01 | Prompt Injection | text in a prompt changes what the model does or says, in ways the app did not intend |
    | LLM02 | Sensitive Information Disclosure | the app lets confidential data out, in a way that the people it is about or its owner did not allow |
    | LLM03 | Excessive Agency | when the model goes wrong, for whatever reason, the app lets it do damaging things |
    | LLM04 | Supply Chain | weak points in what others made: training data, models, the platforms they run on |
    | LLM05 | Data and Model Poisoning | data for training is changed to build in holes, backdoors or biases |
    | LLM06 | Unbounded Consumption | users can make the model work without limits: the service goes down, costs run up, or the model gets copied |
    | LLM07 | Misinformation | the model writes false or misleading information that looks credible |
    | LLM08 | Hidden Context Exposure | someone gets the hidden instructions out of the model, or other context the user should not see |
    | LLM09 | Vector and Embedding Weaknesses | weak points in apps that look things up in documents before they answer |
    | LLM10 | Improper Output Handling | what the model writes goes on to other software without a check |

    The 2025 list has the same ten in another order, with System Prompt Leakage in the place of Hidden Context Exposure.

    ### Vote and data

    From 2025 to 2026, OWASP writes, "Excessive Agency climbed to third", "Unbounded Consumption rose four places", and "Improper Output Handling fell the furthest, from fifth to tenth". Prompt injection stayed first, although the incident data alone put it outside the top 10. OWASP's explanation: "That gap is a defense effect. Teams fight injection hard, so fewer clean exploits reach a public database, and the public count understates the risk that mature teams already spend real money holding off." In the letter at the start of the 2026 list, OWASP gives advice that fits the last lesson, where a small change fooled a model: "Stop trying to build a model that cannot be fooled. Build the system around it, so that when the model is fooled, and it will be, nothing important breaks."

    ### A list for agents

    An **agent**, or LLM agent (the news says AI agent), is an app in which a model takes steps by itself, with tools such as email or a database; [What an agent is: a model, tools and a loop](/course/open/agent-idea?lang=en) builds one. Since December 2025, OWASP has a list for those too, the Top 10 for Agentic Applications, with the ids ASI01 to ASI10: Agent Goal Hijack, Tool Misuse and Exploitation, Identity and Privilege Abuse, Agentic Supply Chain Vulnerabilities, Unexpected Code Execution (RCE), Memory & Context Poisoning, Insecure Inter-Agent Communication, Cascading Failures, Human-Agent Trust Exploitation and Rogue Agents. The first, **Agent Goal Hijack**, starts from the weakness behind prompt injection: "agents and the underlying model cannot reliably distinguish instructions from related content". With an agent, OWASP adds, more is at stake than one answer of the model: text it reads can redirect its goals, its planning and what it does over several steps. An agent is still an app with an LLM, so its review uses both lists.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **LLM** | large language model: a model that reads text and writes text, such as the one behind a chat assistant |
        | **OWASP Top 10 for LLM Applications** | OWASP's list of the ten most critical security risks of apps with an LLM, in editions of 2023, 2025 and 2026 |
        | **edition** | one version of such a list; a finding names it after the id, as in LLM03:2026 |
        | **prompt injection** | text in a prompt that changes what the model does, in ways the app did not intend; LLM01 in every edition |
        | **agent** (LLM agent) | an app in which a model takes steps by itself, with tools such as email or a database |
        | **Agent Goal Hijack** | ASI01, the first entry of OWASP's list for agents: text an agent reads redirects its goal |
        | `int("LLM07"[3:])` | the place of an entry in its list: the characters from the fourth on, as a whole number, here 7 |
        """),
        L.watch_out("A Top 10 is not a certificate. OWASP calls its web list \"a standard awareness document\". An app for which a review answers no to all ten questions can still have a hole that no list names, and the order of a list says what its makers rank highest across many apps, not what matters most in yours."),
        mo.Html(C.idea_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To start the review of an app with OWASP's lists:

    1. Choose the lists that fit: an app with an LLM gets the list for LLM applications; an agent gets the list for agents as well.
    2. Choose the edition, and write it after every id: LLM03:2026 and LLM03:2025 are different entries.
    3. For each entry, ask whether it can happen in your app, and write down what in the app makes it possible.
    4. Write every yes as a finding with its id, edition and name, such as LLM08:2026 Hidden Context Exposure, so that whoever reads it knows which risk is meant.
    5. Then look further: a list starts a review, it does not end it. Build the app so that a fooled model breaks nothing important.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The course keeps the three editions in `C.EDITIONS`: a dict from year to a dict from id to the name of the entry, as in the picture in *The idea*. The cell prints LLM01 and LLM03 of each edition, and works out the place of LLM03 from its id.
    """)
    return


@app.cell
def example_steps():
    # the three editions of OWASP's list: for each year, a dict from id to the name of the entry
    for year, entries in C.EDITIONS.items():
        print(year, "LLM01:", entries["LLM01"], "| LLM03:", entries["LLM03"])
    # the place of an entry is the number in its id, as a whole number
    print("place of LLM03:", int("LLM03"[3:]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Prompt Injection is LLM01 in all three editions; LLM03 is another entry each time. A finding that says only "LLM03" can mean three different risks. `"LLM03"[3:]` takes the characters from the fourth on, `"03"`, and `int` makes the whole number 3 of them.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_where():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. where: one name, every edition

        Write `where(name, editions)`: for each year in `editions`, the id of the entry called `name`, or `None` when that edition has no entry with exactly that name. Give back a dict from year to id. Replace the `...`. The check calls `where("Supply Chain", EDITIONS)` with the three editions of the worked example, then other names and a made-up list of office risks in two editions.
        """),
        L.task("Find the id of name in every edition, and None where the name is missing."),
    ])
    return


@app.cell(hide_code=True)
def exercise_where():
    box_where = L.code_box("where", lines=9)
    box_where
    return (box_where,)


@app.cell(hide_code=True)
def check_where(box_where):
    result_where = L.check("where", box_where.value)
    result_where
    return (result_where,)


@app.cell(hide_code=True)
def hints_where():
    L.hints(
        "where",
        [
            "One loop over the editions, as in the worked example, and inside it a second loop over the entries of that edition.",
            "Give each edition `None` first, and replace it with the id of the entry whose name is exactly `name`.",
            "`for code, entry in entries.items():`, and under it `if ___:` with `found[year] = code`",
        ],
        why="`found = {}`; `for year, entries in editions.items():` with `found[year] = None` and the loop `for code, entry in entries.items():`, in which `if entry == name:` sets `found[year] = code`; then `return found`. `where(\"Supply Chain\", EDITIONS)` gives `{2023: None, 2025: \"LLM03\", 2026: \"LLM04\"}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_where():
    mo.md(r"""
    Supply Chain is LLM03 in 2025 and LLM04 in 2026; in 2023, `where` gives `None`, because that edition called it Supply Chain Vulnerabilities. A program compares names letter by letter, so a renamed entry needs a person who reads what the edition says about it. OWASP did that for System Prompt Leakage: in 2026, it says, that entry is now Hidden Context Exposure.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_moves():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. moves: what rose and what fell

        Write `moves(old, new)`: for each name that is on both lists, how many places it rose from `old` to `new`, negative when it fell. Give back a dict from name to that number. The place of an entry is the number in its id, as in the worked example. Replace the `...`. The check calls `moves(EDITIONS[2025], EDITIONS[2026])`, then other editions and the made-up list.
        """),
        L.task("For each name on both lists: its place on old minus its place on new."),
    ])
    return


@app.cell(hide_code=True)
def exercise_moves():
    box_moves = L.code_box("moves", lines=8)
    box_moves
    return (box_moves,)


@app.cell(hide_code=True)
def check_moves(box_moves):
    result_moves = L.check("moves", box_moves.value)
    result_moves
    return (result_moves,)


@app.cell(hide_code=True)
def hints_moves():
    L.hints(
        "moves",
        [
            "Two loops again: for each entry of `new`, look for an entry of `old` with the same name.",
            "When the names are the same, the rise is the place on the old list minus the place on the new one; the place comes from the id, with `[3:]` and `int`.",
            "`for old_code, old_name in old.items():` inside `for new_code, name in new.items():`, then `risen[name] = int(old_code[3:]) - ___`",
        ],
        why="`risen = {}`; `for new_code, name in new.items():` and inside it `for old_code, old_name in old.items():`, with `if old_name == name:` and `risen[name] = int(old_code[3:]) - int(new_code[3:])`; then `return risen`. Unbounded Consumption, LLM10 in 2025 and LLM06 in 2026, rose 4 places.",
    )
    return


@app.cell(hide_code=True)
def text_after_moves():
    mo.md(r"""
    From 2025 to 2026, Unbounded Consumption rose four places and Excessive Agency three; Improper Output Handling fell five, from fifth to tenth. Those are the moves OWASP describes itself. Your dict has nine names: Hidden Context Exposure is missing, because by its name it is new. From 2023 to 2025, `moves` finds only three names on both lists, Prompt Injection, Sensitive Information Disclosure and Excessive Agency. The other seven names of 2023 are not on the 2025 list as such. Some got a new name, such as Supply Chain Vulnerabilities, and the 2025 list puts some under another entry: Overreliance under Misinformation, and Model Theft under Unbounded Consumption.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_findings():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. findings: a first review

        The library's assistant answers members' questions about the catalog, with an open-weights model from a hub. It can see the loans of every member, and renews and cancels loans by itself, with SQL that it writes and that runs as written. Its hidden instructions hold the password of the catalog's database, and there is no limit on how much a member asks. It names the catalog record each answer comes from; nobody else can add documents to what it looks up, and the library does not train the model further. Because it takes steps by itself, its review needs the list for agents as well, as step 1 of the recipe says; this first review starts with the list for LLM applications, and Exercise 5 comes back to the other list. For each entry of 2026, the course wrote one question, and answered it for the assistant:

        | id | question | answer |
        |---|---|---|
        | LLM01 | Does text that the app did not write reach the model, such as what members type? | yes |
        | LLM02 | Can the model see data that not everyone who talks to it may see? | yes |
        | LLM03 | Can the model do things, such as change data, without a person checking first? | yes |
        | LLM04 | Does the app use a model, data or software that others made? | yes |
        | LLM05 | Does the app train a model further on data that others can change? | no |
        | LLM06 | Can one user ask as much as they like, without a limit? | yes |
        | LLM07 | Do people act on its answers without a source to check them against? | no |
        | LLM08 | Is there anything in its hidden instructions that may not leak? | yes |
        | LLM09 | Does the app look things up in documents that others can add or change? | no |
        | LLM10 | Does its output go on to other software without a check, such as into SQL? | yes |

        `C.ANSWERS` holds those answers: a dict from id to `True` (yes) or `False` (no). Write `findings(answers, entries, edition)`: a list with one line for each entry whose answer is `True`, written as id, colon, edition, a space and the name of the entry, such as `"LLM01:2026 Prompt Injection"`. Replace the `...`. The check calls `findings(ANSWERS, EDITIONS[2026], 2026)`, then the same answers under the ids of 2025, and a second made-up app.
        """),
        L.task("Write a line with id, edition and name for each entry whose answer is True."),
    ])
    return


@app.cell(hide_code=True)
def exercise_findings():
    box_findings = L.code_box("findings", lines=7)
    box_findings
    return (box_findings,)


@app.cell(hide_code=True)
def check_findings(box_findings):
    result_findings = L.check("findings", box_findings.value)
    result_findings
    return (result_findings,)


@app.cell(hide_code=True)
def hints_findings():
    L.hints(
        "findings",
        [
            "Steps 3 and 4 of the recipe: go through the answers, and keep a line for each yes.",
            "Start with an empty list; for each id whose answer is `True`, add an f-string with the id, the edition and the name of the entry, which `entries` gives for that id.",
            "`for code, yes in answers.items():`, and under `if yes:` the line `lines.append(f\"{code}:{edition} {___}\")`",
        ],
        why="`lines = []`; `for code, yes in answers.items():` with `if yes:` and `lines.append(f\"{code}:{edition} {entries[code]}\")`; then `return lines`. For the library's assistant, that gives 7 lines, from `\"LLM01:2026 Prompt Injection\"` to `\"LLM10:2026 Improper Output Handling\"`.",
    )
    return


@app.cell(hide_code=True)
def text_after_findings():
    mo.md(r"""
    Seven of the ten questions get a yes. LLM05 gets a no although the model comes from a hub: a model may have been poisoned before the library got it, and this review counts that under LLM04 Supply Chain, because the library does not train the model further. A yes is not yet a hole: it says where the review looks closer, and the next lessons of this module do that for several entries. *How often does an injection work?* tests LLM01, *Getting the system prompt out* is about LLM08, and *Model output is untrusted input* about LLM10. Under the ids of 2025, several of the same seven findings get other ids: LLM07:2025 System Prompt Leakage where 2026 says LLM08:2026 Hidden Context Exposure. And a no is not a promise: the questions cover the ten entries, not everything that can go wrong.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_vote():
    mo.md(r"""
    ### Exercise 4. Why first?

    Ranked by the incident data alone, prompt injection is not in the top 10 of 2026. OWASP still puts it first.
    """)
    return


@app.cell(hide_code=True)
def choice_vote_cell():
    choice_vote = L.choice(
        "vote",
        question="What reason does OWASP give?",
        options={
            "right": "It is fought hard, so it shows up less in the data",
            "keep": "OWASP never moves the first entry of its list",
            "data": "The incident data counted for nothing in the ranking",
        },
        correct="right",
        why={
            "right": "OWASP: \"That gap is a defense effect. Teams fight injection hard, so fewer clean exploits reach a public database, and the public count understates the risk that mature teams already spend real money holding off.\" A low count can mean a risk that is fought hard, not a small one.",
            "keep": "No such rule: other entries moved as much as five places between 2025 and 2026, and OWASP gives a reason of its own for prompt injection. See *Vote and data* in *The idea*.",
            "data": "The incident data counted for a quarter of the weight: \"The community vote carries three-quarters of the weight. The incident data covers the remaining quarter.\" OWASP explains the gap itself. See *Vote and data* in *The idea*.",
        },
    )
    choice_vote
    return (choice_vote,)


@app.cell(hide_code=True)
def text_exercise_lists():
    mo.md(r"""
    ### Exercise 5. Which lists?

    The library lets its assistant read the email of the help desk, and send email itself, with nobody checking first.
    """)
    return


@app.cell(hide_code=True)
def choice_lists_cell():
    choice_lists = L.choice(
        "lists",
        question="Which of OWASP's lists does its review use?",
        options={
            "right": "Both lists: for LLM applications and for agents",
            "agents": "Only the list for agents: it replaces the other",
            "llm": "Only the list for LLM applications: email is just text",
        },
        correct="right",
        why={
            "right": "The assistant is still an app with an LLM, so every entry of the list for LLM applications still counts. It already changed loans by itself, and now it also reads email that anyone can send to the help desk, and sends email itself, so the list for agents certainly counts, starting with ASI01 Agent Goal Hijack: an email it reads can redirect its goal.",
            "agents": "The list for agents adds the risks of a model that takes steps by itself; the model inside is still an LLM, with every risk of the list for LLM applications. See *A list for agents* in *The idea*.",
            "llm": "Email it reads is text, but the assistant acts on it by itself: it changes loans and sends email. That is what the list for agents is for, starting with ASI01 Agent Goal Hijack. See *A list for agents* in *The idea*.",
        },
    )
    choice_lists
    return (choice_lists,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Attacks on machine learning](/course/open/rai-ml-threat-models?lang=en), a small change fooled a model; OWASP's advice for 2026 starts from the same fact, and asks for a system in which a fooled model breaks nothing important. In [Thinking like an attacker](/course/open/rai-security-basics?lang=en), a threat model asked what can go wrong; a list like OWASP's gives a first set of answers for apps with an LLM. In [Injection and secrets](/course/open/rai-classic-vulnerabilities?lang=en), SQL injection worked because a database cannot tell what the program wrote from what a member typed; prompt injection, LLM01 in every edition, is the same problem with instructions and text. *Poisoning the training data* is next: Data and Model Poisoning, LLM05:2026.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_lists, choice_vote, result_findings, result_moves, result_where):
    L.wrap_up(result_where, result_moves, result_findings, choice_vote, choice_lists)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
