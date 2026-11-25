# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-mt-with-llms. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-mt-with-llms", "en")
    C = L.checks   # the course's twelve texts, its run of Qwen3-0.6B in three ways, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Scoring a translation: BLEU and chrF](/course/open/nlp-bleu?lang=en), and one about [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_order():
    quick_order = L.choice(
        "quick-order",
        question="You put the sentences of a test set in another order, and each translation keeps its own reference. What happens to the BLEU of the whole test set?",
        options={
            "lower": "It goes down, as the n-grams no longer follow each other",
            "right": "It stays the same",
            "penalty": "It goes up, as the brevity penalty changes",
            "far": "It depends on how far each sentence moved from its old place",
        },
        correct="right",
        why={
            "right": "BLEU of a test set adds up the clipped n-gram counts and the lengths of all sentences, and each n-gram is counted within one sentence and its reference. The order of the sentences plays no part, so BLEU cannot see whether a name is written the same way in every sentence of a text. This lesson counts that in another way.",
            "lower": "Each n-gram is counted inside one sentence, against the reference of that sentence. Does an n-gram ever run from one sentence into the next? See [Scoring a translation: BLEU and chrF](/course/open/nlp-bleu?lang=en).",
            "penalty": "The brevity penalty compares the length of all translations with the length of all references. Does that change when the sentences change places? See [Scoring a translation: BLEU and chrF](/course/open/nlp-bleu?lang=en).",
            "far": "Each sentence is still scored against its own reference. What in BLEU looks at where a sentence stands? See [Scoring a translation: BLEU and chrF](/course/open/nlp-bleu?lang=en).",
        },
    )
    quick_order
    return (quick_order,)


@app.cell(hide_code=True)
def quick_check_messages():
    quick_messages = L.choice(
        "quick-messages",
        question="In a chat, you send your third message. What does the model get as its prompt?",
        options={
            "last": "Only your third message",
            "yours": "Your three messages, without its own answers",
            "right": "All messages so far, each with its role",
            "summary": "A summary of the chat so far, which the model wrote itself after each of its answers",
        },
        correct="right",
        why={
            "right": "In a chat, the prompt is every message so far: yours with the role `user`, the model's own answers with the role `assistant`. This lesson uses that to translate a text sentence by sentence in one chat, so that the model sees the earlier sentences and its own translations of them.",
            "last": "Then the model would not know what you asked before. What is a prompt in a chat? See [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en).",
            "yours": "The model's answers are messages too. Which role do they have, and are they in the prompt? See [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en).",
            "summary": "The model gets the messages themselves, each with a role. Which roles are there? See [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en).",
        },
    )
    quick_messages
    return (quick_messages,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    For years, machine translation worked one sentence at a time, and so does Qwen3 in the lesson *Translators and machines together*. But a sentence is often part of a text: a name comes back, and so does a term, and the text around a sentence says which meaning a word has there. In 2018, Samuel Läubli, Rico Sennrich and Martin Volk had professional translators compare Chinese-to-English news translations by a system that was said to translate as well as people with translations by people. Rating single sentences for how well they keep the meaning, the raters found no clear difference. With the whole text, they did: the raters "show a stronger preference for human over machine translation when evaluating documents as compared to isolated sentences." In a news story of six sentences about an app, the human translation kept one name for the app all the way through. Läubli and colleagues write: "In MT, we find three different translations in the same article: “Twitter Move Car”, “WeChat mobile”, and “WeChat Move”."

    A chat model gets its input as a prompt, and a prompt can hold a whole text. This lesson lets Qwen3-0.6B, the small chat model of the course, translate twelve short Dutch texts that the course wrote, in three ways, and counts what happens to the names and the terms that come back.

    **An everyday example.** In a manual, page 2 calls a part the *switch* and page 30 calls it the *button*: are they two parts? Translators of manuals keep a list of terms, so that one word of the original gets one word in the translation, all the way through.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_right = L.guess(
        "guess",
        question="Each text of the course has a term with more than one meaning, such as blad (a leaf, a sheet or a magazine), which comes back three or four times: 42 places in all. With the whole text in one prompt, Qwen3 uses one word for the term in almost every text. In how many of the 42 places does it use a word for what the term names in the text?",
        options={
            "all": "All 42",
            "most": "More than three quarters",
            "between": "Between half and three quarters",
            "right": "Fewer than half",
        },
        reveal="15 of the 42: fewer than half. Each sentence alone, it is 12, and in one chat 14. The whole text makes Qwen3 keep one word for a term far more often, but in six texts that one word does not name what the term names there: bank for the benches of Sluis, forecast for the frost of Best, wool for the fabric of Buren, newspaper for the magazine of Hoorn, rope for the ferry of Velden, and monster for the water sample of Kampen.",
    )
    guess_right
    return (guess_right,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Document-level translation** translates each sentence with the rest of the text as **context**; **sentence-level translation** translates each sentence on its own. A chat model can get the context in two ways. *In one chat*: each sentence is a new message, and the chat keeps all earlier messages, so the model sees the earlier Dutch sentences and its own translations of them. *The whole text*: all sentences in one prompt, each on a numbered line, and the model writes back a numbered line for each. In 2023, Longyue Wang and colleagues tried three such prompts with ChatGPT, each with one chat for the whole text: one sentence per message, or as many sentences per message as possible, with or without a mark between the sentences. The differences were small, and the prompt with many sentences per message and no marks did best on most of their measures, the consistency of terms among them.

    Context helps where a sentence alone can mean two things. In *Het blad verschijnt vier keer per jaar*, *blad* can be a leaf, a sheet or a magazine; the sentence before it says *tijdschrift*, magazine. And many Dutch towns have a name that is also a word: *Bergen* is *mountains*, *Echt* is *real*. Each of the course's twelve texts has such a town, and a term with more than one meaning that comes back three or four times.

    A term that comes back should keep its translation through the whole text. Wang and colleagues measure this as the **consistency** of a term: of all pairs of its translations in one text, the share that are the same. It is 1 when the translation is the same word every time. In their words, "one repeated terminology should keep the same translation throughout the whole document". Consistency does not say whether that word is right: a term translated wrong in the same way every time is consistent too.

    The picture shows one text in the three ways.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    Hoorn stays a name in all three ways. Each sentence alone, Qwen3 has nothing else to go on: *Het blad verschijnt vier keer per jaar* becomes *The leaf appears four times a year*, and in sentence 4 Qwen3 writes *the blade*. In sentences 3 and 5 it leaves the term out. In one chat and with the whole text, Qwen3 writes *newspaper* all four times: no longer a leaf, but something that comes out regularly, and the same word every time. Yet the text is about a magazine that comes out four times a year, and that is not a newspaper.

    Whether a word is right, the course decided by hand for every word Qwen3 used: a word for what the term names in the text is right, also a more general one, such as *building* for the castle of Bergen. A word for another meaning of the term, such as *leaf*, or for another thing, such as *newspaper* for a magazine, is not.
    """)
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **document-level translation** | translating each sentence with the rest of the text as context |
    | **sentence-level translation** | translating each sentence on its own, without the sentences around it |
    | **context** | here: the other sentences of the text that a model sees while it translates one |
    | **consistency** of a term | of all pairs of translations of a term in one text, the share that are the same; 1 when it is the same word every time |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Put each sentence of the text on a line of its own, with its number before it: `1.`, `2.`, and so on.
    2. Give the whole text in one prompt, and ask for a numbered line back for each sentence. Or translate the sentences in one chat, one message each, and keep every earlier message and answer.
    3. Read the numbered lines of the answer back as sentences, and check that every number came back.
    4. For each term, note the word the translation uses in each sentence.
    5. Compute the consistency of each term, and check whether its words are right: consistent is not the same as right.
    6. Score the translations with BLEU against references as well, knowing that BLEU looks at one sentence at a time.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    In the text about Echt, *bord* means a road sign: the first sentence has *verkeersborden*, road signs. Each sentence alone, Qwen3 writes *board* three times and *sign* once. Here are the pairs of those four words, written out by hand, and the share of pairs that are the same:
    """)
    return


@app.cell
def example_cell():
    alone = ["board", "board", "board", "sign"]
    pairs = [(alone[0], alone[1]), (alone[0], alone[2]), (alone[0], alone[3]),
             (alone[1], alone[2]), (alone[1], alone[3]),
             (alone[2], alone[3])]
    same = 0
    for a, b in pairs:
        if a == b:
            same = same + 1
    print(same, "of", len(pairs), "pairs are the same:", same / len(pairs))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Four words make six pairs: the first word with each of the three after it, the second with the two after it, and the third with the last. Three of the six are the same, so the consistency is 0.5. With the whole text, Qwen3 writes *sign* all four times, which gives 1.0: the same word every time, and here also the right one. The course writes each word in the singular, so *boards* counts as *board*. In exercise 3, two loops make the pairs.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_numbered():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. numbered: the lines of the whole text

        Write `numbered(sentences)`, where `sentences` is the list of the sentences of a text. Give back one text with each sentence on a line of its own: its number, counted from 1, a full stop, a space and the sentence. `"\n".join(lines)` joins the texts of the list `lines` with a line break between each two. Replace the `...`. The course puts its instruction for the whole text, `C.DOCUMENT_INSTRUCTION`, and a blank line before your lines: *Translate this Dutch text into English. Translate each numbered line into one numbered line. Reply with only the English translation.*

        The check calls `numbered(["Het regent.", "De weg is glad."])`, which gives `"1. Het regent.\n2. De weg is glad."`. Then cases of its own: an empty list, one sentence, eleven sentences, and the course's twelve texts (in the lesson: `C.DOCUMENTS[i]["nl"]`).
        """),
        L.task("Number the lines of a text for the prompt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_numbered():
    box_numbered = L.code_box("numbered", lines=6)
    box_numbered
    return (box_numbered,)


@app.cell(hide_code=True)
def check_numbered(box_numbered):
    result_numbered = L.check("numbered", box_numbered.value)
    result_numbered
    return (result_numbered,)


@app.cell(hide_code=True)
def hints_numbered():
    L.hints(
        "numbered",
        [
            "In the loop, `k` runs over the places of the sentences, 0, 1, 2 and so on, and `lines.append` adds one line to `lines`: the number of the sentence, a full stop, a space and the sentence `sentences[k]`. An f-string puts the value of what stands between `{` and `}` in the text.",
            "The first sentence is at place 0, but its line should start with 1. What do you write between the braces to get the number of the sentence at place `k`?",
            '`        lines.append(f"{___}. {sentences[k]}")`',
        ],
        why='In the loop: `lines.append(f"{k + 1}. {sentences[k]}")`.',
    )
    return


@app.cell(hide_code=True)
def text_after_numbered():
    mo.md(r"""
    In all twelve texts, Qwen3 gave back one numbered line for each sentence, so the course could put each line next to its Dutch sentence. Keeping to the numbers is not keeping to the task: for the text about Haren, Qwen3 gave back the Dutch sentences themselves, under their numbers. Each sentence alone, it did so only for the first sentence of that text.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chat():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. chat_messages: one chat for a whole text

        Write `chat_messages(instruction, sentences, answers)`. `instruction` is the instruction for one sentence, `sentences` the sentences of the text, and `answers` Qwen3's translations of the first sentences, as many as it has done so far. Give back the messages of the chat for the next sentence: for each sentence so far, a message with the role `"user"` with the instruction, a blank line and the sentence, then Qwen3's answer to it with the role `"assistant"`; and at the end the next sentence with the role `"user"`, in the same form. A message is a dictionary such as `{"role": "user", "content": "Hi"}`, as in *Asking well*. Replace the two `...`.

        The check calls `chat_messages("Translate.", ["Het regent.", "De weg is glad."], ["It is raining."])`, which gives `[{"role": "user", "content": "Translate.\n\nHet regent."}, {"role": "assistant", "content": "It is raining."}, {"role": "user", "content": "Translate.\n\nDe weg is glad."}]`. Then cases of its own: the start of a text with no answers yet, a text of three sentences, and the course's chats for two texts (in the lesson: `C.INSTRUCTION`, `C.DOCUMENTS[i]["nl"]` and `C.answers(i, "chat")[:k]`).
        """),
        L.task("Build the chat in which Qwen3 translates the next sentence."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chat():
    box_chat = L.code_box("chat_messages", lines=8)
    box_chat
    return (box_chat,)


@app.cell(hide_code=True)
def check_chat(box_chat):
    result_chat = L.check("chat_messages", box_chat.value)
    result_chat
    return (result_chat,)


@app.cell(hide_code=True)
def hints_chat():
    L.hints(
        "chat_messages",
        [
            'Each message is a dictionary with a `"role"` and a `"content"`. In the loop, `messages.append` first adds sentence `k` with the role `"user"`, which is there already, and then the translation of that sentence with the role `"assistant"`: `answers` holds Qwen3\'s translations so far. After the loop, one more message with the role `"user"` asks for the next sentence, in the same form as in the loop: `instruction`, a blank line and the sentence.',
            "Qwen3 has translated as many sentences as `answers` holds, and the places in `sentences` start at 0. Which answer goes with sentence `k`, and at which place is the next sentence?",
            '`        messages.append({"role": "assistant", "content": ___})`',
        ],
        why='In the loop: `messages.append({"role": "assistant", "content": answers[k]})`. After it: `messages.append({"role": "user", "content": instruction + "\\n\\n" + sentences[len(answers)]})`.',
    )
    return


@app.cell(hide_code=True)
def text_after_chat():
    mo.md(r"""
    Each sentence alone, *Duiven is de eerste plaats in de buurt met dit vak* becomes *First place in the neighborhood with this subject.*: the town is gone. In the chat, Qwen3's first answer begins with *A school in Duiven*, and it keeps the name. Over the twelve texts, the town stays a name in 22 of its 31 places each sentence alone, and in 29 in one chat. With the whole text, it is 24: Qwen3 gives back the Dutch of the text about Haren, writes *the sluice* twice for Sluis, spells Bergen *Bregen* once, and leaves out Bergen once and Echt once.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_consistency():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. consistency: the same word every time?

        Write `consistency(translations)`, where `translations` is the list of the words a translation used for one term in one text, at least two. Give back the share of the pairs of two different places in the list whose words are the same. Two loops make the pairs: `i` runs over every place, and `j` over the places after `i`. Replace the two `...`.

        The check calls `consistency(["leaf", "paper", "paper"])`, which gives `0.3333`, and `consistency(["menu", "menu", "menu", "menu"])`, which gives `1.0`. Then cases of its own: two translations, the same or not, longer lists, and the words Qwen3 used in the course's run (in the lesson: `C.words_found(i, way)`, without the places where Qwen3 left the term out).
        """),
        L.task("Compute the consistency of a term in one text."),
    ])
    return


@app.cell(hide_code=True)
def exercise_consistency():
    box_consistency = L.code_box("consistency", lines=9)
    box_consistency
    return (box_consistency,)


@app.cell(hide_code=True)
def check_consistency(box_consistency):
    result_consistency = L.check("consistency", box_consistency.value)
    result_consistency
    return (result_consistency,)


@app.cell(hide_code=True)
def hints_consistency():
    L.hints(
        "consistency",
        [
            "The outer loop takes each place `i` of `translations` in turn. The inner loop over `j` should take only the places after `i`, up to the end of the list, so that each pair comes once and no word is paired with itself: `range(a, b)` gives the whole numbers from `a` up to `b`, without `b`, and `len(translations)` is the number of words. The `if` asks whether the two words of the pair are the same.",
            "For three words, the pairs are (0, 1), (0, 2) and (1, 2). At which place does `j` start when `i` is 0, and when `i` is 1?",
            "`        for j in range(___, len(translations)):`",
        ],
        why="`for j in range(i + 1, len(translations)):` and `if translations[i] == translations[j]:`.",
    )
    return


@app.cell(hide_code=True)
def text_after_consistency():
    mo.md(r"""
    On average over the texts, the consistency is 0.54 each sentence alone, 0.92 in one chat and 0.94 with the whole text, that last one over eleven texts, as Qwen3 gave back the text about Haren in Dutch. But in six texts, the whole text has one word every time that does not name what the term names: *bank* for the benches of Sluis, *forecast* for the frost of Best, *wool* for the fabric of Buren, *newspaper* for the magazine of Hoorn, *rope* for the ferry of Velden and *monster* for the water sample of Kampen. The table counts everything.
    """)
    return


@app.cell(hide_code=True)
def table_cell():
    mo.Html(C.table_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.md(r"""
    Context helps Qwen3 with names and with keeping a term the same, and BLEU goes up from 27.4 each sentence alone to 35.9 in one chat and 35.1 with the whole text. It helps far less with the meaning: of the 42 places of a term, 12 get a right word each sentence alone, 14 in one chat and 15 with the whole text. A model that does not know that the Dutch *monster* can be a sample does not learn that from the sentences around it. Wang and colleagues measured ChatGPT, a much larger model; this lesson does not show how much context helps a model of that size.
    """)
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it: every translation of the course's run

    Pick a text and a way, and see each sentence: the Dutch, what Qwen3 wrote, and the course's translation. Look for a town that became an English word, and for a term that changes from sentence to sentence.
    """)
    return


@app.cell(hide_code=True)
def try_controls():
    try_text = mo.ui.dropdown(options=C.text_options(L.lang), value="Hoorn: blad", label="Text")
    try_way = mo.ui.radio(options={C.WAY_NAMES[L.lang][_way]: _way for _way in C.WAYS},
                          value=C.WAY_NAMES[L.lang]["alone"], label="Way")
    mo.hstack([try_text, try_way], justify="start", gap=1.5, wrap=True)
    return try_text, try_way


@app.cell(hide_code=True)
def try_look(try_text, try_way):
    mo.Html(C.try_picture(L.lang, try_text.value, try_way.value))
    return


@app.cell(hide_code=True)
def text_live():
    mo.md(r"""
    ### Translate your own text with Qwen3

    Write a short Dutch text, one sentence per line, and let Qwen3-0.6B translate it in the three ways, the same each time you ask, as in the course's run. Put a town in it whose name is also a word, or a term with more than one meaning, and see which way keeps it. Each line is translated twice, and then the whole text once, so a longer text takes longer. Qwen3 runs on your own computer, so nothing you type leaves it.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1.5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*The chat model Qwen3-0.6B is not on your computer yet. The button downloads it once from Hugging Face, `Qwen/Qwen3-0.6B` at a fixed version, 1.5 GB, under the Apache-2.0 license, with the tokenizer of Qwen3-0.6B-Base. The rest of the lesson works without it: it reads the translations that the course recorded.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: your disk does not have room for 1.5 GB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    mo.md("*Qwen3-0.6B is downloaded.*")
    return


@app.cell(hide_code=True)
def text_box_cell():
    text_box = mo.ui.text_area(value=C.YOUR_TEXT, label="Your Dutch text, one sentence per line", rows=4, full_width=True)
    translate_button = mo.ui.run_button(label="Translate with Qwen3")
    mo.vstack([text_box, translate_button])
    return text_box, translate_button


@app.cell(hide_code=True)
def demo_answer(text_box, translate_button):
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif not translate_button.value:
        _out = mo.md("*Press the button to let Qwen3 translate your text.*")
    elif not C.your_lines(text_box.value):
        _out = mo.md("*Write a text in the box first.*")
    else:
        _out = mo.Html(C.translate_live(text_box.value, lambda _html: mo.output.replace(mo.Html(_html)), L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_same_wrong():
    mo.md(r"""
    ### Exercise 4. Consistent, but right?

    In the text about Kampen, *monster* means a water sample: the first sentence has *watermonster*. In all three ways, every word Qwen3 uses for the term is *monster*, and your `consistency` gives 1.0 for each way.
    """)
    return


@app.cell(hide_code=True)
def choice_same_wrong_cell():
    same_wrong = L.choice(
        "same_wrong",
        question="What does a consistency of 1.0 tell you about the translations of monster in the text about Kampen?",
        options={
            "good": "That the term is translated right, as all three ways agree",
            "right": "That Qwen3 used one word every time, and nothing about whether it is right",
            "bleu": "That BLEU is high for these sentences",
            "broken": "That consistency does not work for a word that is written the same in Dutch and English, so the text should be left out",
        },
        correct="right",
        why={
            "right": "Consistency only compares the translations with each other. Four times *monster* is fully consistent, and four times wrong: a water sample is a *sample*. Elena Voita, Rico Sennrich and Ivan Titov make a related point, on why BLEU hardly tells consistent translations from inconsistent ones: \"For example, if multiple translations of a name are possible, forcing consistency is essentially as likely to make all occurrences of the name match the reference translation as making them all different from the reference.\" So check both: whether a term stays the same, and whether it is right.",
            "good": "All three ways write the same word, but is that what *monster* means in this text? What does the water company take from the river in the first sentence?",
            "bleu": "BLEU compares with the course's translation, which says *sample*. Does *monster* match it?",
            "broken": "Consistency compares Qwen3's words with each other, whatever language they look like. What does it say about whether *monster* is right?",
        },
    )
    same_wrong
    return (same_wrong,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Scoring a translation: BLEU and chrF](/course/open/nlp-bleu?lang=en) you scored a translation one sentence at a time against a reference; here a chat model translates with the sentences around each sentence, in a chat with message roles as in [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en), and consistency counts what BLEU does not look at. The lesson *Translators and machines together* has Qwen3 translate sentence by sentence, and post-edits its translations. The next lesson, *From data to text*, turns data into text, as in a weather report.

    **An everyday example.** Subtitles are translated line by line, but a good subtitler watches the whole film first: who speaks to whom decides whether *you* becomes *je* or *u* in Dutch.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_chat, result_consistency, result_numbered, same_wrong):
    L.wrap_up(result_numbered, result_chat, result_consistency, same_wrong)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
