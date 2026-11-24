# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-extraction-with-llms. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-extraction-with-llms", "en")
    C = L.checks   # the course's notices, its run of Qwen3-0.6B, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [From text to a table](/course/open/nlp-information-extraction?lang=en), and one about [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_score():
    quick_score = L.choice(
        "quick-score",
        question="A rule takes 40 rows out of a text, and 30 of them are right. The gold standard has 50 rows. What are the precision and the recall of the rule?",
        options={
            "swapped": "Precision 0.60, recall 0.75",
            "right": "Precision 0.75, recall 0.60",
            "found": "Precision 0.75, recall 0.80: 40 of the 50 rows were found",
            "same": "Both 0.75",
        },
        correct="right",
        why={
            "right": "Precision is the share of the rows found that are right, 30 / 40 = 0.75; recall is the share of the right rows that were found, 30 / 50 = 0.60. This lesson counts the values a chat model fills in, the same way.",
            "swapped": "Which of the two looks at the rows the rule found, and which at the rows of the gold standard? See [From text to a table](/course/open/nlp-information-extraction?lang=en).",
            "found": "The rule found 40 rows, but only 30 of them are right. Which rows does recall count? See [From text to a table](/course/open/nlp-information-extraction?lang=en).",
            "same": "The two divide the same 30 right rows by different numbers. By which? See [From text to a table](/course/open/nlp-information-extraction?lang=en).",
        },
    )
    quick_score
    return (quick_score,)


@app.cell(hide_code=True)
def quick_check_shots():
    quick_shots = L.choice(
        "quick-shots",
        question="What is few-shot prompting?",
        options={
            "role": "Giving the model a role, such as a teacher",
            "steps": "Asking for the steps before the answer",
            "right": "Putting a few finished examples of the task in the prompt",
            "again": "Asking the same question a few times, and taking the answer that comes back most often",
        },
        correct="right",
        why={
            "right": "Each example is an input with the answer you want, before the real input, so that the model follows the same pattern. This lesson shows a chat model two finished examples of a form to fill in.",
            "role": "That is role prompting. What do the shots of few-shot prompting stand for? See [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en).",
            "steps": "That is chain of thought. What do the shots of few-shot prompting stand for? See [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en).",
            "again": "Asking more than once is a way to check an answer. What do the shots of few-shot prompting stand for? See [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en).",
        },
    )
    quick_shots
    return (quick_shots,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In *From text to a table*, you took the dates out of text with a regular expression: one pattern for each form a date can take. A notice of an event has more fields, and each can be written in many ways: a time is *19:30*, *from 21:00* or *the first round starts at 9:30*, and a place is a building, a street or a park. Writing a pattern for every field and every form is slow work.

    A chat model can fill in the fields from a prompt: say which fields you want, show a few finished examples, and give it the text. Monica Agrawal and others tried this with GPT-3 on notes that doctors wrote, in 2022: "On the clinical extraction tasks we studied, the GPT-3 systems significantly outperform existing zero- and few-shot baselines." But a chat model often writes a value also for a field the text does not give. This lesson lets Qwen3-0.6B, the small chat model of the course, fill in five fields for twelve notices, and checks every value against the notice it came from.

    **An everyday example.** Under an email with a train ticket, your phone shows a button *Add to calendar*: it has read the date, the time and the station from the text. A wrong time in your calendar is worse than no time at all, so a careful app shows what it found before it saves anything.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_made_up = L.guess(
        "guess",
        question="The course wrote the twelve notices of this lesson, and left eight fields out on purpose: one place, four prices, two times and one date. The prompt asks Qwen3 to write - for a field the text does not give, and shows two examples, one of them with a -. How many of those eight fields does Qwen3 fill in anyway, in its answers with seed 0?",
        options={
            "none": "None: the prompt says to write -",
            "few": "One or two",
            "right": "About half",
            "all": "All eight",
        },
        reveal="Five of the eight. Qwen3 wrote - for the price of the cooking workshop, the time of the garden tour and the price of the open day, as the prompt asks. For the other five, it wrote a value: a place for the swimming lessons, a price for Night of the Stars and for the spring cleaning, a time for the film evening and a date for jazz in the park. Over the four seeds, it fills in 23 of the 32.",
    )
    guess_made_up
    return (guess_made_up,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Information extraction** fills the fields of a template from free text. With a chat model, the template goes into the prompt, in the parts of *Asking well*: an **instruction** that names the fields and says how the answer must look, a few **examples** with their finished answers, and then the **text**. Tags such as `<text>` and `</text>` around each text are **delimiters**: they show where a text starts and ends. The model goes on after the last text, and writes one line per field: `date: Friday 20 March`. With examples, that is few-shot prompting; without them, zero-shot prompting.

    The model often writes a value for every field it is asked for, also where the text gives none. A value that is in the text, the model copied. A value that is nowhere in the text came from somewhere else: from an example in the prompt, from what the model saw while it was trained, or from nothing at all. A value that sounds sure but has no source is a **hallucination**. So you **check every extracted value against the source**: you look up each value in the text it should come from, and every value that is not there goes to a person to look at, or stays empty.

    The picture shows one notice, the prompt that Qwen3 got for it, and Qwen3's answer.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    Qwen3 keeps to the form of the examples: one line per field. Four of the values are in the notice. The place is not: the notice does not say where the lessons are, and Qwen3 wrote *the community center*, from the Repair Café in example 1 of the prompt. Looking the value up in the notice catches it, because those words are nowhere in the notice.
    """)
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **hallucination** | output that sounds sure but is false or made up, such as a price that is not in the notice |
    | **source** | the text a value should come from: here, the notice |
    | **checking extracted values** | looking up every value a model filled in in its source, and flagging the values that are not there |
    | `a in b`, `a not in b` | for two strings: `True` when the text `a` does, or does not, occur somewhere in `b` |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Write the instruction: the fields, with a few words on each, the form of the answer, and what to write for a field the text does not give.
    2. Add two or three examples with their finished answers, one of them with a field left out.
    3. Put the text between tags after them.
    4. Read the answer line by line into a row of fields, and leave out the fields with `-`.
    5. Look up every value in the text, and flag the values that are not there.
    6. Let a person look at the flagged values, or leave those fields empty.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The notice of the swimming lessons says when they start and what they cost, but not where they are. Here is the row that Qwen3's answer gives, with the two examples and seed 0, and each value looked up in the notice by hand:
    """)
    return


@app.cell
def example_cell():
    notice = "Swimming lessons for adults start on Monday 16 March at 19:30. Ten lessons cost €60."
    row = {"what": "Swimming lessons", "date": "Monday 16 March", "time": "19:30",
           "place": "the community center", "price": "€60"}
    for field in row:
        print(field, "|", row[field], "|", row[field].lower() in notice.lower())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Four values print `True`: they are in the notice. *Swimming lessons* is shorter than the name in the notice, but the words are there. The place prints `False`: *the community center* is not in the notice, but in example 1 of the prompt. Both the value and the notice go to lower case first, so that *Monday* and *monday* count as the same. In exercise 3, you write this check as a function for a whole row.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_prompt():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. fields_prompt: the prompt

        Write `fields_prompt(instruction, examples, text)`. `instruction` is a text, `examples` a list of tuples of a text and its finished answer, and `text` the text to fill in the fields for. Give back one text: the instruction and a blank line, then for each example `<text>` on a line of its own, the text of the example, `</text>` on a line of its own, the answer and a blank line, and at the end `<text>`, the text and `</text>`, each on a line of its own. `"\n"` in a string is a line break, so `"\n\n"` makes a blank line. Replace the two `...`.

        The check calls `fields_prompt("Fill in two fields: what and date.", [("Quiz in the café on Friday 6 March.", "what: Quiz\ndate: Friday 6 March")], "Bingo on Monday 9 March.")`, which gives `"Fill in two fields: what and date.\n\n<text>\nQuiz in the café on Friday 6 March.\n</text>\nwhat: Quiz\ndate: Friday 6 March\n\n<text>\nBingo on Monday 9 March.\n</text>"`, and the same in Dutch. Then cases of its own: no examples, and the course's instruction and two examples for Night of the Stars, in English and in Dutch (in the lesson: `C.INSTRUCTION`, `C.SHOTS` and `C.NOTICES`).
        """),
        L.task("Build the prompt that asks for the fields."),
    ])
    return


@app.cell(hide_code=True)
def exercise_prompt():
    box_prompt = L.code_box("fields_prompt", lines=9)
    box_prompt
    return (box_prompt,)


@app.cell(hide_code=True)
def check_prompt(box_prompt):
    result_prompt = L.check("fields_prompt", box_prompt.value)
    result_prompt
    return (result_prompt,)


@app.cell(hide_code=True)
def hints_prompt():
    L.hints(
        "fields_prompt",
        [
            "In the loop, `prompt` grows by one example at a time: the text of the example between the two tags, then its `answer`. After the loop, `return` gives back `prompt` with the `text` between the tags at the end.",
            'Strings join with `+`: `"<text>\\n" + example` puts the tag on its own line, before the text. What comes after the text, and what after the answer?',
            "`    return prompt + ___`",
        ],
        why='In the loop: `prompt = prompt + "<text>\\n" + example + "\\n</text>\\n" + answer + "\\n\\n"`. After it: `return prompt + "<text>\\n" + text + "\\n</text>"`.',
    )
    return


@app.cell(hide_code=True)
def text_after_prompt():
    mo.md(r"""
    Without examples, Qwen3 often copies the words of the instruction into its answer: for Night of the Stars, *time: when it starts* and *price: of a ticket or entry*, and for other notices one line *place and price:* with both in it. Of the 48 answers without examples (twelve notices, four seeds), 13 have a line `field: value` for each of the five fields; with the two examples, 44 do. And without examples, Qwen3 does not write `-` once in the 48 answers. The examples show the form, where the instruction only describes it, and example 2 shows a `-`. They make the prompt longer: for Night of the Stars, 274 tokens instead of 123.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_read():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. read_fields: the answer as a row

        Write `read_fields(answer)`, where `answer` is the text that the model wrote. Give back a dictionary from each field to its value, for every part of the answer in the form field, a colon, a space and a value up to the end of the line, such as `date: Friday 20 March`. Take the spaces off both ends of each value with `.strip()`: Qwen3 often ends a line with two spaces. Leave out a field whose value is `-`. A field name is made of word characters, `\w`. Replace the two `...`: the pattern, with a group for the field and a group for the value, and the lines in the loop.

        The check calls `read_fields("what: Bingo  \ndate: Monday 9 March  \nprice: -")`, which gives `{"what": "Bingo", "date": "Monday 9 March"}`, and the same in Dutch. Then cases of its own: an empty answer, and answers of Qwen3 from the course's run, with and without examples (in the lesson: `C.answer(lang, way, i)["text"]`).
        """),
        L.task("Read the model's answer into a row of fields."),
    ])
    return


@app.cell(hide_code=True)
def exercise_read():
    box_read = L.code_box("read_fields", lines=10)
    box_read
    return (box_read,)


@app.cell(hide_code=True)
def check_read(box_read):
    result_read = L.check("read_fields", box_read.value)
    result_read
    return (result_read,)


@app.cell(hide_code=True)
def hints_read():
    L.hints(
        "read_fields",
        [
            "`re.findall` with a pattern of two groups gives a tuple `(field, value)` for every match in `answer`. In the loop, `value.strip()` takes off the spaces, and `if value != \"-\":` puts the value in `row` under its `field`.",
            "A group is a part of the pattern between round brackets. `\\w+` matches a field name, and `.` any character except a line break: what matches the rest of a line after `: `?",
            '`    for field, value in re.findall(r"(\\w+): ___", answer):`',
        ],
        why='The pattern is `r"(\\w+): (.+)"`. In the loop: `value = value.strip()`, then `if value != "-":` with `row[field] = value` under it.',
    )
    return


@app.cell(hide_code=True)
def text_after_read():
    mo.md(r"""
    Without examples, the answer for the chess tournament has the line *time (when it starts): 9:30*. There, no word is followed straight by `: `, so the time is lost. And *place and price: Sports hall on School Lane, €5, players under 12 free* becomes a price with the place in it: the pattern takes the last word before the colon as the field. A row is only as good as the form of the answer: of the twelve answers of seed 0 without examples, your function reads all five fields from two; with the examples, from eight, and three more have a `-` where the notice gives nothing.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_flag():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. unsupported: values that are not in the source

        Write `unsupported(row, text)`, where `row` is a dictionary of fields and values, as `read_fields` gives it, and `text` the source. Give back a list of the fields whose value is not in the text, in the order of the row. Capitals should not matter: compare both in lower case, with `.lower()`. Replace the `...`.

        The check calls `unsupported({"what": "Bingo", "date": "Monday 9 March", "price": "free"}, "Bingo on Monday 9 March.")`, which gives `["price"]`, and the same in Dutch. Then cases of its own: an empty row, a row in other capitals, and rows from Qwen3's answers with their notices (in the lesson: `C.read_fields(C.answer(lang, way, i)["text"])` and `C.NOTICES[lang][i][0]`).
        """),
        L.task("Flag the values that are not in the source."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flag():
    box_flag = L.code_box("unsupported", lines=8)
    box_flag
    return (box_flag,)


@app.cell(hide_code=True)
def check_flag(box_flag):
    result_flag = L.check("unsupported", box_flag.value)
    result_flag
    return (result_flag,)


@app.cell(hide_code=True)
def hints_flag():
    L.hints(
        "unsupported",
        [
            "In the loop, `value.lower()` is the value in lower case. With `if ... not in ...:` you ask whether it is missing from the text, also in lower case, and then `fields.append(field)` adds the field.",
            "`not in` works on two strings: is the one nowhere in the other? Which of the two should be found in which, and what do both need, so that capitals do not count?",
            "`        if value.lower() not in ___:`",
        ],
        why="`if value.lower() not in text.lower():` with `fields.append(field)` under it.",
    )
    return


@app.cell(hide_code=True)
def text_after_flag():
    mo.md(r"""
    Your function flags one value of the 56: *the community center* as the place of the swimming lessons. But Qwen3 filled in five of the eight fields that the notices leave out, so four wrong values get through: *bring warm clothes*, *a silent film with live piano music*, *gloves and bags* and *14:00*. Over the four seeds, Qwen3 fills in 23 of the 32, and the check flags 5 of them. It flags no value that is right: when Qwen3 copies, as the instruction asks, the value is in the notice.
    """)
    return


@app.cell(hide_code=True)
def text_long():
    mo.md(r"""
    ### A long document

    A real text often holds more than one event. Here all twelve notices are in one newsletter of 1,440 characters, and the prompt ends with a line that names the event: *Fill in the fields for this event only: Cooking workshop*. The values now have to come from one part of a longer text. With seed 0, Qwen3 gives the cooking workshop a price of €5, the entry of the chess tournament, and the garden tour a time of 18:00, the time of the cooking workshop. A check against the whole newsletter finds those values in it, and lets them through: over the four seeds, it flags 2 values, where a check against the notice of the event flags 24. So look up each value in the part of the text it should come from, not in the whole document. The table counts all three ways.
    """)
    return


@app.cell(hide_code=True)
def table_cell():
    mo.Html(C.table_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it: every answer of the course's run

    Pick a way of asking, a seed and a notice, and see the prompt, what Qwen3 wrote, and every value looked up in the notice. With the newsletter, each value is also looked up in the whole newsletter. Look for a value that the check lets through, though it is wrong.
    """)
    return


@app.cell(hide_code=True)
def try_controls():
    try_way = mo.ui.radio(options={"without examples": "zero", "with the two examples": "examples", "with the newsletter": "long"},
                          value="with the two examples", label="Way of asking")
    try_seed = mo.ui.slider(0, 3, value=0, label="Seed", show_value=True)
    try_notice = mo.ui.dropdown(options=C.notice_options(L.lang), value="2. Swimming lessons for adults", label="Notice")
    mo.hstack([try_way, try_seed, try_notice], justify="start", gap=1.5, wrap=True)
    return try_notice, try_seed, try_way


@app.cell(hide_code=True)
def try_look(try_notice, try_seed, try_way):
    mo.Html(C.try_picture(L.lang, try_way.value, try_seed.value, try_notice.value))
    return


@app.cell(hide_code=True)
def text_live():
    mo.md(r"""
    ### Ask Qwen3 yourself

    Write a notice of your own, and let Qwen3-0.6B fill in the five fields, with the instruction and the two examples of the course and seed 0. Leave out a field, such as the price, and see whether Qwen3 writes `-` or fills in something else. Qwen3 runs on your own computer, so nothing you type leaves it.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1.5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*The chat model Qwen3-0.6B is not on your computer yet. The button downloads it once from Hugging Face, `Qwen/Qwen3-0.6B` at a fixed version, 1.5 GB, under the Apache-2.0 license, with the tokenizer of Qwen3-0.6B-Base. The rest of the lesson works without it: it reads the answers that the course recorded.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: your disk does not have room for 1.5 GB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    mo.md("*Qwen3-0.6B is downloaded.*")
    return


@app.cell(hide_code=True)
def notice_box_cell():
    notice_box = mo.ui.text_area(value=C.YOUR_NOTICE[L.lang], label="Your notice", rows=3, full_width=True)
    ask_button = mo.ui.run_button(label="Ask Qwen3")
    mo.vstack([notice_box, ask_button])
    return ask_button, notice_box


@app.cell(hide_code=True)
def demo_answer(ask_button, notice_box):
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif not ask_button.value:
        _out = mo.md("*Press the button to ask Qwen3.*")
    elif not notice_box.value.strip():
        _out = mo.md("*Write a notice in the box first.*")
    else:
        _out = mo.Html(C.ask_live(notice_box.value, lambda _html: mo.output.replace(mo.Html(_html)), L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_passes():
    mo.md(r"""
    ### Exercise 4. A value that gets through

    In the picture under exercise 3, Qwen3's answer for Night of the Stars, with the two examples and seed 0, has *price: bring warm clothes*. The notice gives no price at all, and still `unsupported` does not flag the value.
    """)
    return


@app.cell(hide_code=True)
def choice_passes_cell():
    why_passes = L.choice(
        "why_passes",
        question="Why does unsupported let bring warm clothes through as the price of Night of the Stars?",
        options={
            "made_up": "Qwen3 made the words up, and a check on the text cannot see that a value is made up",
            "capitals": "The check compares in lower case, and that changes the value",
            "right": "The words are in the notice, and the check does not look at the field",
            "long": "The notice is too long for the check",
        },
        correct="right",
        why={
            "right": "The notice ends with *Bring warm clothes.*, so the value is in the text. A check that a value is in its source finds values that come from nowhere or from an example, such as *the community center*, but not words of the source in the wrong field. For those you need a check per field, such as a price that must hold € or *free*, or a person who looks.",
            "made_up": "Look up the words in the notice of Night of the Stars: are they there? What does `unsupported` ask about a value?",
            "capitals": "Lower case only makes *Bring* and *bring* the same. Are the words of the value in the notice?",
            "long": "The notice has three sentences, and `in` looks through all of it. Are the words of the value in the notice?",
        },
    )
    why_passes
    return (why_passes,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [From text to a table](/course/open/nlp-information-extraction?lang=en) you wrote one pattern for each form of a date, and here a chat model fills in five fields from a prompt in the parts of [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en); your `read_fields` still reads its answer with a regular expression. Why a model writes a value that has no source, [Why language models make things up](/course/open/lm-hallucination?lang=en) explains. The next lesson, *Kinds of question answering*, asks questions about a text instead of filling in fields: where the answer comes from, the text or the model, matters there too.

    **An everyday example.** A shop that reads the invoices of its suppliers with a model checks the total against the invoice, number for number, before it pays: an amount that is on no invoice is never paid.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_flag, result_prompt, result_read, why_passes):
    L.wrap_up(result_prompt, result_read, result_flag, why_passes)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
