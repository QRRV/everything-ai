# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-slot-filling. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-slot-filling", "en")
    C = L.checks   # the booking system with its frame, the conversation, the ticket commands of MASSIVE and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [From text to a table](/course/open/nlp-information-extraction?lang=en), and one about [ELIZA: a chatbot from rules](/course/open/nlp-eliza?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_groups():
    quick_groups = L.choice(
        "quick-groups",
        question=r'In *From text to a table* you took dates out of text with groups. What does `re.findall(r"(\d+) (May|June)", "on 28 May or 4 June")` give?',
        options={
            "whole": '`["28 May", "4 June"]`: each whole match, day and month together',
            "first": '`("28", "May")`: what the two groups matched the first time',
            "right": '`[("28", "May"), ("4", "June")]`: a pair for each match',
            "days": '`["28", "4"]`: what the first group matched',
        },
        correct="right",
        why={
            "right": "With two groups in the pattern, `re.findall` gives a pair for each match, with one string for each group. In this lesson, each slot has a pattern with one group, and the system needs the first match only.",
            "whole": "The pattern has groups. What does `re.findall` give when a pattern has more than one group? See [From text to a table](/course/open/nlp-information-extraction?lang=en).",
            "first": "Those are the groups of the first match. Does `re.findall` stop at the first match? See [From text to a table](/course/open/nlp-information-extraction?lang=en).",
            "days": "The pattern has two groups, one around the day and one around the month. What does `re.findall` give for each match? See [From text to a table](/course/open/nlp-information-extraction?lang=en).",
        },
    )
    quick_groups
    return (quick_groups,)


@app.cell(hide_code=True)
def quick_check_eliza():
    quick_eliza = L.choice(
        "quick-eliza",
        question="In *ELIZA: a chatbot from rules*, ELIZA answered by its script of 1965. A user types *My train leaves at nine.*, and ELIZA answers *YOUR TRAIN LEAVES AT NINE*. Then the user types *What time does my train leave?* What does ELIZA answer?",
        options={
            "nine": "*AT NINE*: it looks up what the user said about the train",
            "memory": "*YOUR TRAIN LEAVES AT NINE*, its memory of line 1",
            "right": "*WHY DO YOU ASK*: a reassembly rule of *what*",
            "none": "*PLEASE GO ON*: the line has no keyword",
        },
        correct="right",
        why={
            "right": "*What* and *my* are both keywords of rank 0, so ELIZA takes the first one, *what*, and a reassembly rule of *what* gives *WHY DO YOU ASK*. ELIZA kept no train time: its memory holds sentences to say back, not facts to look up. The system of this lesson keeps what the user said, in slots.",
            "nine": "ELIZA answers by the rules of a keyword in the new line. Does it keep facts about what the user said? See [ELIZA: a chatbot from rules](/course/open/nlp-eliza?lang=en).",
            "memory": "ELIZA says a memory only when a line has no keyword and its counter is at 4. Does this line hold a keyword? See [ELIZA: a chatbot from rules](/course/open/nlp-eliza?lang=en).",
            "none": "*What* and *my* are keywords of the script. Which one does ELIZA pick, and what do its rules give? See [ELIZA: a chatbot from rules](/course/open/nlp-eliza?lang=en).",
        },
    )
    quick_eliza
    return (quick_eliza,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In 1977, six researchers at the Xerox Palo Alto Research Center, Daniel Bobrow, Ronald Kaplan, Martin Kay, Donald Norman, Henry Thompson and Terry Winograd, described GUS, a program that played a travel agent for a simple trip by air. In their words: "GUS (Genial Understander System) is intended to engage a sympathetic and highly cooperative human in an English dialog". Dan Jurafsky and James Martin printed a transcript of a conversation with GUS in an earlier draft of their book *Speech and Language Processing*. It starts like this:

    > GUS: Hello. My name is GUS. I can help you plan a simple trip by air. Where do you want to go ?
    > Client: I want to go to San Diego on May 28
    > GUS: What time do you want to leave?
    > Client: I must be in San Diego before 10 am
    > GUS: Would you like P.S.A. flight 102 that arrives at 9.15 am ?

    The client gave two things in one sentence, a city and a date, and GUS did not ask for the date again: it asked for what it still lacked. ELIZA answered each line on its own; GUS kept track of what the client had said, in a form with an empty place for each thing it needed. The client never named a city to leave from, yet a confirmation later in the conversation says *from San Jose to San Diego*: GUS filled that place by itself.

    Jurafsky and Martin write that GUS "in the late 1970s established the frame-based paradigm that became the dominant industrial paradigm for dialogue systems for over 30 years", and that "By around 2010 the GUS architecture finally began to be widely used commercially in dialogue systems on phones like Apple's SIRI (Bellegarda, 2013) and other digital assistants." In this lesson you build such a system for train tickets, and run its first step on real commands to an assistant.

    **An everyday example.** Say *Set an alarm* to a phone, and it asks *For what time?*: an alarm needs a time, and the phone asks for what you left out.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_frame = L.guess(
        "guess",
        question="MASSIVE is a dataset of commands to an assistant, which Jack FitzGerald and colleagues at Amazon published in 2022, in 51 languages with Dutch among them; in each command, the words of its slots are marked. 187 of its commands ask for a ticket, such as *book me a train ticket*. A command that fills the whole frame of this lesson at once names two places, where from and where to, and a date. In how many of the 187 does MASSIVE mark at least two places and a date?",
        options={
            "most": "140, three out of four",
            "half": "93, about half",
            "quarter": "47, a quarter",
            "right": "8",
        },
        reveal="8 of the 187, such as *buy a train ticket for one week from now from new york to las vegas nevada*. MASSIVE marks at least two places without a date in 16 more, one place in 98, and no place in 65; in 55 of those 65, it marks neither a place nor a date, as in *book me a train ticket*. Most commands leave something open, so a system that needs three values has to ask for the rest.",
    )
    guess_frame
    return (guess_frame,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    A **dialogue system** is a program that holds a conversation with a person over several turns. ELIZA was one without a task. A system for **task-oriented dialogue** helps the user with one concrete task, such as booking a ticket, and it is done when the task is. Jurafsky and Martin, in the appendix *Frame-Based Dialogue Systems* of their book: "Frames are knowledge structures representing the details of the user's task specification. Each frame consists of a collection of slots, each of which can take a set of possible values."

    A **frame** is a form for a task, and a **slot** is a place in it for one thing the task needs. In the appendix: "An alarm clock system needs a time." The frame of this lesson has three slots, in this order: *to*, the station to travel to; *from*, the station to leave from; and *date*, the day of the trip. Each slot has a question, taken from the frame for air travel in the appendix: *Where are you going?*, *From what city are you leaving?* and *What day would you like to leave?*. In Jurafsky and Martin's words: "The system's goal is to fill the slots in the frame with the fillers the user intends, and then perform the relevant action for the user (answering a question, or booking a flight)."

    For each line the user types, the system of this lesson takes these steps:

    1. **Slot filling.** Finding the values for the slots in what the user says is **slot filling**. "The simplest dialogue systems use handwritten rules for slot-filling", such as this regular expression from the appendix for a request to set an alarm: `wake me (up) | set (the|an) alarm | get me up`. The system of this lesson writes the line in small letters and has one regular expression for each slot, with one group: `\bto (atlanta|boston|…|washington)\b` for *to*, the same with *from* for *from*, and the days of the week with *today* and *tomorrow* for *date*. The stations are in the pattern, so *want to go to boston* gives *boston*, not *go*. A slot whose expression matches gets what the group matched, the first time.
    2. **The frame.** The values found go into their slots, and the other slots keep what they had. Jurafsky and Martin: "The dialogue-state is not just the slot-fillers in the current sentence; it includes the entire state of the frame at this point, summarizing all of the user's constraints." Keeping that state up to date after each turn is called dialogue state tracking.
    3. **The next question.** Jurafsky and Martin: "In early commercial frame-based systems, the dialogue policy is simple: ask questions until all the slots are full, do a database query, then report back to the user." The system asks the question of the first slot that is still empty.
    4. **A bare answer.** When a line holds no value at all, the system reads it again with the word of the slot it just asked about in front of it: after *From what city are you leaving?*, *New York.* is read as *from new york.*
    5. **Confirmation.** After a turn that fills a slot, the question starts with what the system understood so far, as in *A train to Boston on Friday. From what city are you leaving?*: an **implicit confirmation**, which, in the appendix, "grounds more implicitly, for example by repeating the system's understanding as part of asking the next question". When every slot is filled, the system says the whole trip and asks *Shall I book it?*: an **explicit confirmation**. Only a yes books the ticket; after a no, the user says what to change. The appendix weighs the two: "Explicit confirmation makes it easier for users to correct misrecognitions by just answering "no" to the confirmation question. But explicit confirmation is time-consuming and awkward".

    **Intents.** An assistant does more than one task, so it first decides which task a command asks for, such as buying a ticket or setting an alarm. That kind of request is the **intent** of the command, and finding it is **intent detection**: a classifier with one label for each intent. The system of this lesson does one task only, so every line has the same intent. In MASSIVE, each command has one of 60 intents; the 187 ticket commands have the intent `transport_ticket`, with their slots marked, as in `book a [transport_type : train] ticket to [place_name : new york]`.

    The picture follows turn 1 of the course's conversation through the steps.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **dialogue system** | a program that holds a conversation with a person over several turns |
    | **task-oriented dialogue** | a conversation in which the system helps the user with one concrete task, such as booking a ticket |
    | **frame** | a form for a task, with a slot for each thing the task needs |
    | **slot** | a place in a frame for one value, such as the station to travel to |
    | **slot filling** | finding the values for the slots in what the user says |
    | **intent** | the kind of request a command makes, such as buying a ticket; finding it is **intent detection** |
    | **implicit confirmation** | saying what the system understood as part of its next question |
    | **explicit confirmation** | asking the user whether what the system understood is right, such as *Shall I book it?* |
    | `.group(1)` | on a match of `re.search`: the text that the first group of the pattern matched |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To book a ticket with a frame:

    1. Start with a frame in which every slot is empty, and ask the question of the first slot.
    2. For each line, look for the value of each slot with its regular expression: what the group matches the first time.
    3. Found nothing? Read the line again with the word of the slot you asked about in front of it.
    4. Put the values found into the frame; every other slot keeps its value.
    5. Ask the question of the first empty slot, with what you understood in front of it.
    6. When no slot is empty, say the whole trip and ask whether to book it; book only on a yes.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Take turn 1 of the conversation. `C.CONVERSATION` holds the four lines of the user in the language of this lesson, and the system first writes a line in small letters with `.lower()`. `C.SLOTS` holds the names of the slots in the order of the frame, and `C.PATTERNS` the regular expression of each slot, for each language. `re.search` looks for the first match of each expression. Then `C.find_slots` gives the values found, `C.new_frame()` a frame in which every slot is empty, `C.fill` the frame after the turn, and `C.next_slot` the first slot that is still empty:
    """)
    return


@app.cell
def example_cell():
    import re

    _text = C.CONVERSATION[L.lang][0].lower()
    print(_text)
    for _slot in C.SLOTS:
        print(_slot, re.search(C.PATTERNS[L.lang][_slot], _text))
    _found = C.find_slots(_text, C.PATTERNS[L.lang])
    _frame = C.fill(C.new_frame(), _found)
    print(_found)
    print(_frame)
    print(C.next_slot(_frame))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The expression of *to* matches *to boston*: in the printed match, `span=(13, 22)` says that it runs from place 13 up to, not including, place 22 of the line, and `match='to boston'` gives its text. Its group holds *boston*. The *to* in *want to go* is not followed by a station, so it does not match there. The expression of *from* finds nothing, and `re.search` gives `None`. The expression of *date* matches *friday*. After the turn, the frame holds *boston* and *friday*, and *from* is still empty, so the system answers *A train to Boston on Friday. From what city are you leaving?*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_find_slots():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. find_slots: filling slots from a line

        Write `find_slots`, which gets `text`, a line in small letters, and `patterns`, a dictionary from each slot to its regular expression with one group, such as `C.PATTERNS["en"]`. It gives back a dictionary with, for each slot whose expression matches, what the group matched the first time; a slot without a match stays out of it. In the start code, `import re` makes `re` available in the box, `found` starts as an empty dictionary, and the loop goes through each `slot`; `m = re.search(patterns[slot], text)` is the first match, or `None` when the expression does not match. Replace the `...`: when `m` is a match, the value goes into `found`.

        The check calls `find_slots("i want to go to boston on friday", C.PATTERNS["en"])`, which gives `{"to": "boston", "date": "friday"}`, and `find_slots("ik wil vrijdag naar utrecht", C.PATTERNS["nl"])`, with the expressions of the Dutch lesson, which gives `{"to": "utrecht", "date": "vrijdag"}`. Then cases of its own: an empty line, all three slots in one line, a station of two words, two stations after *to*, a Dutch line with all three slots, two slots of another frame (a color and a thing), and no expressions at all.
        """),
        L.task("Give each slot whose regular expression matches the text of its group, the first time."),
    ])
    return


@app.cell(hide_code=True)
def exercise_find_slots():
    box_find_slots = L.code_box("find_slots", lines=10)
    box_find_slots
    return (box_find_slots,)


@app.cell(hide_code=True)
def check_find_slots(box_find_slots):
    result_find_slots = L.check("find_slots", box_find_slots.value)
    result_find_slots
    return (result_find_slots,)


@app.cell(hide_code=True)
def hints_find_slots():
    L.hints(
        "find_slots",
        [
            "`found[slot] = x` puts `x` into the dictionary `found`, under the name of the slot. On a match, `.group()` gives the whole text that the match covers.",
            "When the regular expression does not match, `m` is `None`, without groups: first test with `m is not None` whether there is a match. The value of a slot is the station or the day alone, without *to* or *from* in front of it: `.group(1)` gives the text of the first group.",
            "`            found[slot] = ___`",
        ],
        why="`if m is not None:` with `found[slot] = m.group(1)` inside it: the text of the group, under the name of the slot, only when there is a match.",
    )
    return


@app.cell(hide_code=True)
def text_after_find_slots():
    mo.md(r"""
    The picture runs `find_slots` on the ticket commands of MASSIVE whose places are all stations of the lesson. With the course's version, it finds 55 of the 65 places and dates that MASSIVE marks in the 43 English commands, and 77 of the 88 in the 58 Dutch ones. It misses what its expressions do not know: *ticket for new york*, with neither *to* nor *from*; *phoenix to los angeles*, where the station to leave from has no *from*; and dates without a day of the week, such as *april fourteenth* and *two days from now*. Rules written by hand do well on what their writer thought of and miss the rest. In the appendix, right after the regular expression for the alarm: "But most systems use supervised machine-learning": they learn slot filling from marked examples, as a label for each word; the lesson *A label per word: sequence labeling and BIO* shows how such labels work.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fill():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. fill: keeping the frame up to date

        Write `fill`, which gets `frame`, the frame so far as a dictionary from each slot to its value (`None` while the slot is empty), and `found`, the values that `find_slots` found in this turn. It gives back a new frame with the same slots, in the same order: a slot that this turn filled gets its new value, and every other slot keeps the value it had. In the start code, `new` starts as an empty dictionary and the loop goes through each `slot` of `frame`; a slot that is in `found` already gets its new value. Replace the `...`.

        The check calls `fill({"to": None, "from": None, "date": None}, {"to": "boston", "date": "friday"})`, which gives `{"to": "boston", "from": None, "date": "friday"}`, and `fill({"to": "utrecht", "from": None, "date": "vrijdag"}, {"from": "groningen"})`, from the Dutch lesson, which gives `{"to": "utrecht", "from": "groningen", "date": "vrijdag"}`. Then cases of its own: nothing found, a new value for a slot that was filled already, three new values at once, two empty slots of a Dutch frame filled at once, two slots of another frame, and a frame without slots.
        """),
        L.task("Give the new values of this turn, and keep the value of every other slot."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fill():
    box_fill = L.code_box("fill", lines=9)
    box_fill
    return (box_fill,)


@app.cell(hide_code=True)
def check_fill(box_fill):
    result_fill = L.check("fill", box_fill.value)
    result_fill
    return (result_fill,)


@app.cell(hide_code=True)
def hints_fill():
    L.hints(
        "fill",
        [
            "`new[slot] = x` puts `x` into the new frame under the name of the slot.",
            "A slot that this turn did not fill keeps its old value: the value of the same slot in `frame`, the frame from before the turn.",
            "`            new[slot] = ___`",
        ],
        why="`new[slot] = frame[slot]`: the value that the slot already had.",
    )
    return


@app.cell(hide_code=True)
def text_after_fill():
    mo.md(r"""
    Because a new value takes the place of an old one, the user can correct the system: *No, on Saturday.* changes the date and nothing else. Because every other slot keeps its value, the user never has to say a thing twice. The frame after each turn is what the appendix calls the dialogue state; larger systems keep more in it, such as the dialogue act of the user's last turn: whether it was a question, an answer or a correction.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_next_slot():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. next_slot: the next question

        Write `next_slot`, which gets `frame` and gives back the first slot, in the order of the frame, that is still empty, or `None` when every slot is filled. In the start code, the loop goes through each `slot` of `frame` and gives it back at once when the test in the `if` holds; after the loop, `return None` says that no slot is empty. Replace the `...` with that test.

        The check calls `next_slot({"to": "boston", "from": None, "date": "friday"})`, which gives `"from"`, and `next_slot({"to": "utrecht", "from": "groningen", "date": "vrijdag"})`, which gives `None`. Then cases of its own: a new frame with every slot empty, two empty slots, only the last slot empty, two slots of another frame, and a frame without slots.
        """),
        L.task("Give the first slot of the frame that is still empty, or None."),
    ])
    return


@app.cell(hide_code=True)
def exercise_next_slot():
    box_next_slot = L.code_box("next_slot", lines=6)
    box_next_slot
    return (box_next_slot,)


@app.cell(hide_code=True)
def check_next_slot(box_next_slot):
    result_next_slot = L.check("next_slot", box_next_slot.value)
    result_next_slot
    return (result_next_slot,)


@app.cell(hide_code=True)
def hints_next_slot():
    L.hints(
        "next_slot",
        [
            "`frame[slot]` is the value of the slot `slot` in the frame: a station or a day, or `None` while the slot is empty.",
            "The slot to ask about is one whose value is still the `None` of a new frame. Test the value of the slot, not its name, the way your find_slots tested `m` against `None`.",
            "`        if frame[slot] ___:`",
        ],
        why="`if frame[slot] is None:`: the slot has no value yet.",
    )
    return


@app.cell(hide_code=True)
def text_after_next_slot():
    mo.md(r"""
    The order of the frame is the order of the questions: GUS asked where to first, and so does this system. A user who says more than was asked skips questions: turn 1 of the course's conversation fills two slots at once. After turn 2, no slot is empty, `next_slot` gives `None`, and the system says the whole trip and asks *Shall I book it?*. *No, on Saturday.* holds a value, so the system fills it and asks again; *Yes, please.* holds no value and starts with *yes*, so the system books.

    ### Try it: book a ticket

    Type one line per turn in the box. The system answers each line, and the picture shows the frame after each turn. Try a line with two slots at once, a bare answer, a correction after a *no*, and a station the system does not know. Which lines does it get wrong, and does a confirmation let you catch it?
    """)
    return


@app.cell(hide_code=True)
def try_write():
    try_text = mo.ui.text_area(value=C.TRY_START[L.lang], label="Your lines", rows=6, full_width=True)
    try_text
    return (try_text,)


@app.cell(hide_code=True)
def try_look(try_text):
    mo.Html(C.try_picture(L.lang, try_text.value))
    return


@app.cell(hide_code=True)
def text_exercise_what():
    mo.md(r"""
    ### Exercise 4. What does the system fill in?

    The system has just said its greeting and asked *Where are you going?*. Sort each answer by what the system of this lesson fills in: only right values, a slot with a wrong value, or no slot at all. A value that the user gave but the system misses does not make a filled slot wrong.
    """)
    return


@app.cell(hide_code=True)
def exercise_what():
    sort_what = L.sort("what")
    sort_what
    return (sort_what,)


@app.cell(hide_code=True)
def check_what(sort_what):
    result_what = L.check("what", sort_what)
    result_what
    return (result_what,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [From text to a table](/course/open/nlp-information-extraction?lang=en) took fields out of text with regular expressions with groups; slot filling does the same in a conversation, with one field for each slot. [ELIZA: a chatbot from rules](/course/open/nlp-eliza?lang=en) answered each line on its own; the system of this lesson keeps a frame across the turns, asks for what is missing and knows when the task is done. Its confirmations are the grounding of [How a conversation works](/course/open/nlp-conversation?lang=en): saying what was understood, so that the other can correct it. The next lessons of the domain are about search: finding the documents that answer a question.

    **An everyday example.** A ticket machine at a station asks the same things on its screen, one after the other: where to, which day, and whether to pay. Where from is the station the machine stands in, unless you change it: a slot that is filled before you start, as GUS filled in San Jose.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_find_slots, result_fill, result_next_slot, result_what):
    L.wrap_up(result_find_slots, result_fill, result_next_slot, result_what)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
