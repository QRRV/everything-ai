# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-conversation. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-conversation", "en")
    C = L.checks   # the course's conversation, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [What is meant, not said](/course/open/nlp-pragmatics?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_salt():
    quick_salt = L.choice(
        "quick-salt",
        question="What does a listener usually hear in *Can you pass the salt?*",
        options={
            "right": "A request to pass the salt",
            "ability": "A question about whether you are able to",
            "promise": "A promise that the speaker will pass the salt soon",
        },
        correct="right",
        why={
            "right": "The form is a question about ability, and the act is a request: an indirect speech act. This lesson labels such acts turn by turn, in a whole conversation.",
            "ability": "That is what the words ask, taken literally. A listener who only answers *yes* has missed what was meant: the speaker wants the salt. See *Saying is doing* in [What is meant, not said](/course/open/nlp-pragmatics?lang=en).",
            "promise": "A promise binds the speaker, and here the speaker wants something from the listener. See *Saying is doing* in [What is meant, not said](/course/open/nlp-pragmatics?lang=en).",
        },
    )
    quick_salt
    return (quick_salt,)


@app.cell(hide_code=True)
def quick_check_quantity():
    quick_quantity = L.choice(
        "quick-quantity",
        question="*Some of the guests came.* Why does a listener take it that not all of them came?",
        options={
            "right": "The speaker could have said *all*, and did not",
            "logic": "In logic, *some* means not all",
            "presupposition": "It is a presupposition: it stays when you add *not*",
        },
        correct="right",
        why={
            "right": "That is the maxim of Quantity at work: a cooperative speaker who knew that all came would have said so. The *not all* is an implicature, and this lesson keeps assuming that speakers cooperate.",
            "logic": "In logic, *some* means at least one, also when all came. The *not all* is what a listener works out on top of it. See *Some, but not all* in [What is meant, not said](/course/open/nlp-pragmatics?lang=en).",
            "presupposition": "A presupposition is taken for granted. The *not all* is something a listener works out, and the speaker can take it back: *some of the guests came; in fact, all of them did*. It is an implicature. See *Some, but not all* in [What is meant, not said](/course/open/nlp-pragmatics?lang=en).",
        },
    )
    quick_quantity
    return (quick_quantity,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A voice assistant, a help desk program or a language model that you chat with all take part in a conversation, and a conversation has an order of its own. Jurafsky and Martin list six features of conversation that are among the reasons it is hard for a program to take part as people do: turns, speech acts, grounding, how a conversation is built up, who takes the lead, and implicature. A program that answers one sentence well can still get the conversation wrong: it talks while you are thinking, or misses that your *yeah* answered its question.

    **An everyday example.** You ask a voice assistant for a train to ... and stop for a moment to think of the town. It already answers, before you said where to. Or you finish, and it stays so quiet that you ask whether it heard you.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Tanya Stivers and colleagues (2009) measured, in video recordings of everyday conversations in ten languages, Dutch among them, how soon a response to a yes/no question starts after the question ends. About how long was that on average?",
        options={
            "right": "About 0.2 seconds",
            "second": "About 1 second",
            "think": "About 2 seconds: time to think of the answer",
        },
        reveal="**About 0.2 seconds**: 208 milliseconds on average over all ten languages, and the most common value was 0, a response that starts right as the question ends. Danish speakers were the slowest on average (469 ms), Japanese speakers the fastest (7 ms). The means of all ten languages lay within about 250 ms of the overall mean, \"approximately the length of time it takes to produce a single English syllable\", the authors write. They looked at responses to yes/no questions only, from 350 questions per language, and rounded every time to 100 ms.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Taking turns

    A conversation is not a text with one writer: two or more people make it together. For Jurafsky and Martin, a dialogue "is a sequence of turns", "each a single contribution from one speaker to the dialogue". A **turn** is such a contribution. The sociologists Harvey Sacks, Emanuel Schegloff and Gail Jefferson (1974) listed facts about conversation. Their list, quoted here from a reprint that calls itself a variant version of the paper of 1974, starts: "Speaker change recurs, or, at least, occurs", "Overwhelmingly, one party talks at a time", and talk by more than one speaker at a time is "common, but brief". Most changes of speaker come with no gap and no overlap, or with a slight one. They wrote that these facts seemed "grossly apparent to relatively unmotivated examination of conversational materials".

    How speakers share the talk in this way, one after the other, is **turn-taking**. In Jurafsky and Martin's words: "most of the time in conversation, speakers start their turns almost immediately after the other speaker finishes, without a long pause". The guess showed how short that is. To measure it, write down for each stretch of talk who speaks, from when and to when. At a change of speaker, the start of the new turn minus the end of the one before is the **gap**. A positive gap is a silence; a negative one is an **overlap**, the new speaker starting before the old one has stopped. Stivers and colleagues measured the same way. A pause between two stretches of the same speaker is no gap between turns: that turn goes on.
    """)
    return


@app.cell(hide_code=True)
def text_idea_pairs():
    mo.md(r"""
    ### Turns in pairs

    Some turns come in pairs. Emanuel Schegloff and Harvey Sacks (1973) described the **adjacency pair**: two utterances, one right after the other, by different speakers, such as question and answer, a greeting and a greeting back, or an offer and its acceptance or refusal. Pairs "properly have" these features, they wrote: it is what speakers go by, not a law without exceptions. Once the first part has been said, in their words, "its speaker should stop and a next speaker should start and produce a second pair part".

    Jurafsky and Martin add that the two parts of a pair can be separated by a side sequence. An example of the course's own: *Want a coffee?* *Is there oat milk?* *Yes.* *Then yes, please.* The question about the milk and its answer sit inside the pair of the offer and its acceptance.
    """)
    return


@app.cell(hide_code=True)
def text_idea_grounding():
    mo.md(r"""
    ### Making sure it landed

    A speaker cannot look inside the listener's head. The psychologist Herbert Clark and Edward Schaefer (1989) split a contribution in two: the speaker presents an utterance, and the listener accepts it by giving evidence that, as far as the listener knows, it was understood. They list five kinds of evidence, "graded roughly from weakest to strongest": continued attention; starting the relevant next contribution; an acknowledgment, such as a nod or "uh huh" or "yeah"; a demonstration of what was understood; and a display, repeating "verbatim all or part" of what was said.

    Clark and Susan Brennan (1991) call the whole process **grounding**: speaker and listener work toward the shared belief that the listener has understood what was meant "to a criterion sufficient for current purposes". A train time asks for more care than small talk. Acknowledgments, they write, are much of "what has been called back-channel responses". A **backchannel** is a short signal such as *mhm* or *yeah* that lets the other speaker go on.
    """)
    return


@app.cell(hide_code=True)
def evidence_picture():
    mo.Html(C.evidence_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_acts():
    mo.md(r"""
    ### What a turn does

    *What is meant, not said* called an utterance, seen as an act, a speech act. A program that takes part in a conversation, or studies one, gives each turn such an act from a fixed list of labels: a **dialogue act**, in Jurafsky and Martin's words "a tag which represents the interactive function of the sentence being tagged". It works like the parts of speech in *Parts of speech and phrases*, with a label for each turn instead of for each word.

    In 1997, eight students of linguistics labeled 1,155 telephone conversations of the Switchboard collection by hand: 205,000 utterances and 1.4 million words. They used 220 tags, which the researchers grouped into 42 classes for their experiments. The largest two: statements, 36 in 100 utterances, and backchannels and acknowledgments, 19 in 100; statements of opinion were a class of their own, with 13 in 100. Yes/no questions were 2 in 100. Andreas Stolcke and colleagues (2000) then trained a program that labels each utterance. On their test conversations, working from written transcripts, it got 71 in 100 right, where always saying *statement* gets 35 in 100. Two people who labeled the same utterances agreed on 84 in 100.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **turn** | one contribution of one speaker to a conversation |
        | **turn-taking** | how speakers share the talk, one after the other |
        | **gap** and **overlap** | at a change of speaker: the silence between the two turns, or the time both speak at once; here the start of the new turn minus the end of the old |
        | **adjacency pair** | two turns by different speakers in which the first makes the second expected, such as question and answer |
        | **grounding** | the work speaker and listener do to reach the shared belief that what was meant has been understood |
        | **backchannel** | a short signal such as *mhm* or *yeah* that lets the other speaker go on |
        | **dialogue act** | what a turn does, as a label from a fixed list, such as statement, question or backchannel |
        | `"?" in text` | `True` when the text contains a question mark: `in` looks inside a text as it looks inside a list |
        """),
        L.watch_out("In lessons about language models that answer from documents, **grounding** means something else: making a model base its answer on sources you give it. In this lesson it is what two people in a conversation do to make sure they understood each other."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Put the stretches of talk in order of time: who speaks, from when, to when. Stretches of the same speaker in a row are one turn.
    2. At each change of speaker, measure the gap: the start of the new turn minus the end of the one before. Below zero is an overlap.
    3. Look for pairs: does a turn answer, accept or return the turn before it?
    4. Look for grounding: how does the listener show what was understood, from *mhm* to repeating the words?
    5. Give each turn a dialogue act: what does it do in the conversation?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first cell writes a conversation of the course's own: Sam asks Kim about the last train. `TALK` holds each stretch of talk as who speaks, from when and to when, in milliseconds; `SAID` holds the words of each stretch, in the same order. The second cell writes a list of backchannels. The third defines two functions, tries them on `TALK` and looks at the first two stretches.
    """)
    return


@app.cell
def talk_cell():
    TALK = [
        ("Sam", 0, 1500), ("Kim", 1700, 3000), ("Sam", 2950, 3500), ("Kim", 3600, 3900), ("Sam", 3900, 5200),
        ("Sam", 5800, 6500), ("Kim", 6600, 6900), ("Sam", 6950, 8400), ("Kim", 8700, 9000), ("Sam", 9100, 9600),
    ]
    SAID = [
        "When does the last train to Utrecht leave?", "At 11:40, from platform 4", "Platform 4?", "Yeah",
        "So I have to leave home at eleven", "or a bit before", "Mhm", "Could you send me the times?", "Okay",
        "Thanks",
    ]
    return SAID, TALK


@app.cell
def backchannels_cell():
    BACKCHANNELS = ["mhm", "uh-huh", "yeah", "right", "okay"]
    return (BACKCHANNELS,)


@app.cell
def example_look(BACKCHANNELS, SAID, TALK):
    def speakers(talk):
        return [stretch[0] for stretch in talk]

    def lengths(talk):
        out = []
        for i in range(len(talk)):
            out.append(talk[i][2] - talk[i][1])
        return out

    print(speakers(TALK))
    print(lengths(TALK))
    # from the end of the first stretch to the start of the second, and the words of the second
    print(TALK[1][1] - TALK[0][2], SAID[1])
    print("?" in SAID[0], SAID[3] in BACKCHANNELS)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `speakers` takes place 0 of each stretch, and `lengths` gives how long each stretch lasts: its end minus its start. Kim starts 200 milliseconds after Sam stops. `TALK[4]` and `TALK[5]` are both Sam's: one turn, with a pause in it. The last line asks whether the first turn has a question mark, and it has, and whether *Yeah* is in `BACKCHANNELS`: `False`, since the list writes its words in small letters. Exercise 2 has to deal with that.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_gaps():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. gaps: the time between turns

        Write `gaps(talk)`: for each change of speaker in `talk`, give the gap in milliseconds, the start of the new stretch minus the end of the one before, in the order of the conversation. A pause between two stretches of the same speaker is no gap: leave it out. For `TALK`, the first change gives `200`, and the second gives `-50`: Sam starts *Platform 4?* before Kim has finished. Replace the `...`. The check calls `gaps(TALK)`, then cases of its own, with a single stretch, with an overlap and with a pause within a turn.
        """),
        L.task("Measure the gap or overlap at each change of speaker."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gaps():
    box_gaps = L.code_box("gaps", lines=6)
    box_gaps
    return (box_gaps,)


@app.cell(hide_code=True)
def check_gaps(box_gaps):
    result_gaps = L.check("gaps", box_gaps.value)
    result_gaps
    return (result_gaps,)


@app.cell(hide_code=True)
def hints_gaps():
    L.hints(
        "gaps",
        [
            "Start with an empty list `out`, and go over the places from 1 with `for i in range(1, len(talk)):`, as `lengths` goes over them from 0. Add to `out` only where the speaker changes.",
            "Place 0 of a stretch holds the speaker. For which pairs of stretches in a row does a new turn begin?",
            "`        if ___:`",
        ],
        why="`out = []`; `for i in range(1, len(talk)):` with `if talk[i][0] != talk[i - 1][0]:` under it and `out.append(talk[i][1] - talk[i - 1][2])` under that; after the loop `return out`. `gaps(TALK)` gives `[200, -50, 100, 0, 100, 50, 300, 100]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_gaps():
    mo.md(r"""
    The picture under the check draws the conversation as two rows of stretches, Sam's and Kim's, with your numbers at each change of speaker. Most gaps are a few hundred milliseconds or less, and one is an overlap. The longest silence, the 600 milliseconds between *at eleven* and *or a bit before*, is no gap at all: Sam keeps the turn. A listener who took that pause as the end would have cut in.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: when is Sam done?

    A program that listens has to decide when your turn is over. The simplest way is to wait for a silence of a fixed length, and then start talking. Below, Sam asks a program for a train, with two pauses to think. Choose how long the program waits for silence, and see where it starts to talk. Look for a wait at which it does not cut in, and see how long Sam then waits for an answer.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_wait = mo.ui.slider(100, 1500, step=50, value=300, label="Wait for silence (ms)", show_value=True)
    demo_wait
    return (demo_wait,)


@app.cell(hide_code=True)
def demo_look(demo_wait):
    mo.Html(C.demo_picture(demo_wait.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    To let Sam think for 900 milliseconds, the program has to wait longer than that after every stretch, also after the last one. People do better: they respond to a question about as fast as the guess showed, and Kim lets Sam's pause of 600 milliseconds go by. Stivers and colleagues sum up earlier work on English: "speakers do not wait for pauses to begin their turn but avoid gaps and overlaps. To achieve this they use grammar, prosody, and pragmatics to project when they can start a next turn". Prosody is the melody and rhythm of speech. A program that only listens for silence has none of that, and has to pick one wait for both cases.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_label():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. label: a dialogue act for each turn

        Write `label(text, backchannels)`: give back a dialogue act for the turn `text`, by three rules, in this order. A turn that is in the list `backchannels`, whatever its capitals, is a `"backchannel"`; a turn with a question mark in it is a `"question"`; every other turn is a `"statement"`. Replace the `...`. The check calls `label("Mhm", BACKCHANNELS)`, then cases of its own, each with a list of its own.
        """),
        L.task("Label a turn as backchannel, question or statement."),
    ])
    return


@app.cell(hide_code=True)
def exercise_label():
    box_label = L.code_box("label", lines=6)
    box_label
    return (box_label,)


@app.cell(hide_code=True)
def check_label(box_label):
    result_label = L.check("label", box_label.value)
    result_label
    return (result_label,)


@app.cell(hide_code=True)
def hints_label():
    L.hints(
        "label",
        [
            "Write the three rules in order, each as an `if` with its own `return`: first the backchannel, then the question. The last line, after both, gives back `\"statement\"`.",
            "The list holds its words in small letters, and *Mhm* starts with a capital. What can you do to the turn before you look it up in the list?",
            "`    if ___ in backchannels:`",
        ],
        why="`if text.lower() in backchannels:` with `return \"backchannel\"` under it; `if \"?\" in text:` with `return \"question\"` under it; at the end `return \"statement\"`. `label(\"Mhm\", BACKCHANNELS)` gives `'backchannel'`.",
    )
    return


@app.cell(hide_code=True)
def text_after_label():
    mo.md(r"""
    The picture under the check gives every turn of the conversation your label, the label of the three rules and a label by hand, the course's own, in the style of Switchboard. The rules see only the form of a turn. *Yeah* after *Platform 4?* is an answer, *Could you send me the times?* is a request, and *Okay* accepts it. To see that, a program needs the turn before it, and what a speaker does with words.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_evidence():
    mo.md(r"""
    ### Exercise 3. How does Sam show it landed?

    It is eight o'clock at the station. Kim tells Sam: *Your train leaves at eight fifteen, from platform 7.* Put each bold reply of Sam under what it shows of what Sam understood.
    """)
    return


@app.cell(hide_code=True)
def exercise_evidence():
    sort_evidence = L.sort("evidence")
    sort_evidence
    return (sort_evidence,)


@app.cell(hide_code=True)
def check_evidence(sort_evidence):
    result_evidence = L.check("evidence", sort_evidence)
    result_evidence
    return (result_evidence,)


@app.cell(hide_code=True)
def text_exercise_yeah():
    mo.md(r"""
    ### Exercise 4. Answer or backchannel?

    One word can do different things in a conversation. The rules of exercise 2 call every *yeah* a backchannel.
    """)
    return


@app.cell(hide_code=True)
def choice_yeah_cell():
    choice_yeah = L.choice(
        "yeah",
        question="In which exchange is *Yeah* an answer, and not a backchannel?",
        options={
            "right": "*Sam: Is the café still open? Kim: Yeah.*",
            "story": "*Sam: So I got to the station, Kim: Yeah, Sam: and the train had gone.*",
            "plan": "*Sam: I'll take the bus to the center, Kim: Yeah, Sam: and walk the last part from the bridge.*",
        },
        correct="right",
        why={
            "right": "*Is the café still open?* is a question: the first part of a pair, which makes an answer the expected next turn. Kim's *Yeah* is that answer. The rules of exercise 2 call it a backchannel all the same: they look at the turn alone, not at the turn before it.",
            "story": "Sam is in the middle of a story, and Kim's *Yeah* says: go on, I am listening. That is a backchannel. Look for a *Yeah* right after a question.",
            "plan": "Sam is telling a plan in two parts, and Kim's *Yeah* between them says: go on. That is a backchannel. Look for a *Yeah* right after a question.",
        },
    )
    choice_yeah
    return (choice_yeah,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [What is meant, not said](/course/open/nlp-pragmatics?lang=en) looked at single utterances: what a speaker means and does with words. This lesson put them in a conversation: turns that follow each other with short gaps, pairs such as question and answer, grounding that shows what was understood, and a dialogue act for each turn. You measured gaps and overlaps, labeled turns by three rules, sorted kinds of evidence of understanding and told an answer from a backchannel. Later lessons build programs that take turns in a conversation themselves, starting with ELIZA, from 1966. The lesson *A text is more than its sentences* looks at how the sentences of a text hang together.

    **An everyday example.** On the phone, you say *mhm* now and then without thinking about it. Stay silent for a while, and the other person asks whether you are still there.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_yeah, result_evidence, result_gaps, result_label):
    L.wrap_up(result_gaps, result_label, result_evidence, choice_yeah)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
