# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson start-prompt-chains. Route: course/route/lessen/bouwen.yaml. Checks, the four experiments and the course's run: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("start-prompt-chains", "en")
    C = L.checks   # the four experiments, how the lesson reads an answer, and the course's run of Qwen3
    R = C.course()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_delimit():
    quick_delimit = L.choice(
        "quick-delimit",
        question="You paste a customer's email into a prompt, and ask a model to sum it up. Why put the email between `<email>` and `</email>`?",
        options={
            "speed": "So the model reads it faster",
            "apart": "So the model can tell your instruction from the email",
            "needed": "Because a model cannot read an email without tags",
        },
        correct="apart",
        why={
            "speed": "Tags add a few tokens; they do not make reading faster. What if the email itself says *Please reply today*? See [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en).",
            "apart": "The tags are delimiters: a sentence in the email, such as *Please reply today*, is read as part of the text, not as an instruction to the model.",
            "needed": "Without tags the model still reads the email; it just cannot tell as well where your instruction ends. See [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en).",
        },
    )
    quick_delimit
    return (quick_delimit,)


@app.cell(hide_code=True)
def quick_check_last_line():
    quick_last_line = L.choice(
        "quick-last-line",
        question="You want a model to work out a sum, and a program reads the result. How do you end the prompt?",
        options={
            "only": "*Reply with only the number.*",
            "first": "*Give the answer first, then your steps.*",
            "last": "*Work it out step by step. Then give the final answer on the last line, as: Answer: &lt;number&gt;*",
        },
        correct="last",
        why={
            "only": "With only the number, the model has no room to work the sum out before it writes the answer. See the table of the seven ways in [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en).",
            "first": "Once the answer is written, the steps that come after it can no longer change it. See [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en).",
            "last": "The steps come first, where the model works it out, and the answer comes on a fixed last line, where a program can find it.",
        },
    )
    quick_last_line
    return (quick_last_line,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model's answer reads well, whether it is right or not. And one prompt that asks for a lot at once gives the model many places to slip, without showing you where. Three habits help. Split a big task into small prompts, one after the other, so you can look at every answer on the way. Let the model check its own answer, or check each claim with a question of its own. And give the model the text to answer from, with the request to name where it found each fact, so you can look it up. This lesson tries each habit on Qwen3-0.6B, the small chat model of [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en), eight times each, and shows what each fixes, and what it does not.
    """)
    return


@app.cell(hide_code=True)
def predict_refine():
    guess_refine = L.guess(
        "refine",
        question=C.refine_question(L.lang),
        options={
            "most": "Most of them",
            "some": "A few of them",
            "none": "None of them",
        },
        reveal=C.refine_reveal(L.lang),
    )
    guess_refine
    return (guess_refine,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Prompt chaining** splits a task into steps, each its own prompt: the answer of one goes into the next. Each prompt asks less, and between two steps you, or your program, can look at the answer so far.

    A model can also look at its own answer. **Self-Refine** asks it, in a second prompt, to check its answer and improve it. **Chain-of-Verification** goes further: first a draft answer; then, for each claim in it, a separate question in a fresh prompt, without the draft, so the model does not simply agree with what it wrote; and then a final answer with only the claims that passed.

    A model can also answer from a text you give it, instead of from what it remembers: that is **grounding**. Ask it to name the sentence that each part of its answer comes from, a **citation**, and you can check the answer against the text in a moment. Ask it as well to say so when the text does not answer the question.

    A prompt can hold more than text. A **multimodal prompt** has a picture, a sound or a file in it too, for a model that can read those: a photo of a receipt, with the question what it all cost. Qwen3-0.6B reads only text, so this lesson keeps to text.

    ### New words and symbols

    | Word | What it means | Example |
    |---|---|---|
    | **prompt chaining** | splitting a task into small prompts in a row, where the answer of each goes into the next | first the amounts, then the cost |
    | **Self-Refine** | letting the model check its own answer in a second prompt, and improve it | *Check your answer step by step. If it is wrong, fix it.* |
    | **Chain-of-Verification** | a draft answer, then a fresh question for each claim in it, then a final answer with only the claims that passed | *Is 101 a prime number between 80 and 100?* |
    | **grounding** | letting a model answer from a text you give it, not from what it remembers | *Answer using only the text between the source tags.* |
    | **citation** | the place in the given text that a part of the answer comes from | *[2]* |
    | **multimodal prompt** | a prompt with a picture, a sound or a file in it, as well as text | a photo of a receipt and *What did this cost?* |
    """)
    return


@app.cell(hide_code=True)
def picture_habits():
    mo.vstack([
        draw.flow(
            [("Prompt 1", ["the amounts", f"for {C.PEOPLE} people"]), ("Prompt 2", ["the amounts of answer 1,", "with the prices"]),
             ("Answer 2", ["the cost,", "on its last line"])],
            ["answer 1:\nthe amounts", "answer 2"], ["", ""], title="Prompt chaining: one task in two prompts"),
        draw.flow(
            [("A draft", ["five primes", "between 80 and 100"]), ("A check per number", ["Is 89 prime and", "between 80 and 100? …", "each in a fresh prompt"]),
             ("The final answer", ["only the numbers", "that passed"])],
            ["each number\nin the draft", "yes or no\nfor each"], ["", ""], title="Chain-of-Verification"),
        mo.md(r"""
        Between every two boxes, there is an answer you can look at: the amounts before they are priced, each number before it goes into the list. The four experiments below are these two pictures, Self-Refine and grounding, on Qwen3-0.6B. The prompts are in English, as in [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en), because Qwen3-0.6B writes English best.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_chain():
    mo.md(rf"""
    ### Experiment 1. One prompt or a chain

    The task: *{C.RECIPE} I make pancakes for {C.PEOPLE} people. {C.PRICES} How much do the ingredients cost?* The answer is {C.COST} euros. In one prompt, the model has to scale the recipe and price it at once. In a chain, prompt 1 asks only for the amounts, and prompt 2 gets those amounts with the prices. Both ask for the steps first, and the answer on the last line. The course ran both {R['seeds']} times.
    """)
    return


@app.cell(hide_code=True)
def chain_cell():
    L.explore("chain-runs", "run", lambda _v: C.chain_frame(int(_v), L.lang), start=C.CHAIN_SHOWN, low=1, high=R["seeds"], step=1, role="",
              task="Go through the runs. When the chain goes wrong, in which prompt does it go wrong, and could you have seen that before prompt 2?")
    return


@app.cell(hide_code=True)
def text_chain_after():
    mo.md(C.chain_after(L.lang))
    return


@app.cell(hide_code=True)
def text_refine():
    mo.md(rf"""
    ### Experiment 2. Checking its own answer

    The course took Qwen3's first answers to the three sums of [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en), asked as they are and with *Reply with only the final number*, and gave each back to Qwen3 with one more message: *{C.markdown_safe(C.CHECK)}* The table counts the answers that were right before the check and after it.
    """)
    return


@app.cell(hide_code=True)
def text_refine_table():
    mo.md(C.refine_table(L.lang) + "\n\n" + C.refine_after(L.lang))
    return


@app.cell(hide_code=True)
def text_verify():
    mo.md(rf"""
    ### Experiment 3. Chain-of-Verification

    The draft: *{C.draft_primes()[0]['content']}* There are only three primes between {C.LOW} and {C.HIGH}: {', '.join(str(n) for n in sorted(C.PRIMES)[:-1])} and {max(C.PRIMES)}, so a draft of five has to be wrong somewhere. Then, for each number in the draft, a fresh prompt, in two ways. Plainly: *{C.verify(101)[0]['content']}* And worked out first: *{C.verify_steps(101)[0]['content']}* The final list keeps the numbers that got a yes. The course ran it {R['seeds']} times.
    """)
    return


@app.cell(hide_code=True)
def verify_cell():
    L.explore("verify-runs", "run", lambda _v: C.verify_frame(int(_v), L.lang), start=C.VERIFY_SHOWN, low=1, high=R["seeds"], step=1, role="",
              task="Go through the runs. Which mistakes of the draft do the checks catch, and which do they miss?")
    return


@app.cell(hide_code=True)
def text_verify_after():
    mo.md(C.verify_after(L.lang))
    return


@app.cell(hide_code=True)
def text_source():
    mo.md(rf"""
    ### Experiment 4. Answering from a source

    Three questions, {R['seeds']} times each: *{C.ASK_SATURDAY}* without a source; the same with the opening hours of the library as four numbered sentences, and the request to answer only from them and name the sentence; and *{C.ASK_MAIL}*, with the same source, which says nothing about that.
    """)
    return


@app.cell(hide_code=True)
def source_cell():
    L.explore("source-runs", "run", lambda _v: C.source_frame(int(_v), L.lang), start=C.SOURCE_SHOWN, low=1, high=R["seeds"], step=1, role="",
              task="Go through the runs. Where does the answer come from in each of the three, and which ones can you check?")
    return


@app.cell(hide_code=True)
def text_source_after():
    mo.md(C.source_after(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To get answers you can check:

    1. Split a task with several steps into prompts, one step each, and hand each answer on to the next prompt.
    2. Look at each answer between two steps, or let your program check it, before it goes on.
    3. For a claim that has to be right, ask a question about it on its own, in a fresh prompt; let the model work the answer out, and read its steps, not just its yes.
    4. When there is a text with the answer in it, give it between delimiters, and ask the model to answer only from it.
    5. Ask it to name the sentence of each fact, and to say so when the text does not answer the question.
    6. Check the citations: does that sentence say that?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A café wants a short weekly newsletter with this week's specials. The cook writes them in a long note to the staff, with prices, stock and who works when. One prompt, *Write a newsletter from this note*, may pick up a wrong price, or a dish that is no longer on the menu. The recipe, step by step:

    1. Two prompts. Prompt 1: *List the specials in the note between the `<note>` tags, each with its price, one per line.* Prompt 2: the newsletter, from that list.
    2. Between the two, the café reads the list: five dishes, five prices. Here one dish is missing, so it fixes the list before prompt 2.
    3. A price must be right, so for each dish a fresh question: *What does the note say the pea soup costs?*, with the note. Where the answer and the list differ, the café looks at the note.
    4. Prompt 2 gets the list between `<specials>` tags: *Write the newsletter using only the dishes and prices between the tags.*
    5. Prompt 1 also asks for the line of the note each dish comes from, and *If the note has no price for a dish, write: no price.*
    6. The café checks two lines at random: does that line of the note say that?

    No step needs a better model. Each one gives the café a place to see a mistake before a customer does.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Which habit is it?

        Six ways people use a model. Sort them: which of the four is each?
        """),
        L.task("Put each of the six in its group."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_habits = L.sort("habits")
    sort_habits
    return (sort_habits,)


@app.cell(hide_code=True)
def check_sort(sort_habits):
    result_habits = L.check("habits", sort_habits)
    result_habits
    return (result_habits,)


@app.cell(hide_code=True)
def hints_sort():
    L.hints(
        "habits",
        [
            "Ask of each: are there several prompts that each do one step, does the model judge its own answer, is a claim asked about on its own, or does the model answer from a text it was given?",
            "Two of the six hand the answer of one prompt to the next, and two let the model look at what it wrote itself. Which one asks about each claim on its own, in a new chat?",
            "The outline and the dates in the calendar are chains; the one with the manual is grounding; the years asked one by one are Chain-of-Verification.",
        ],
        why="The outline first and then the talk, and the dates first and then the calendar, are prompt chaining: each prompt does one step. Reading its answer again and judging its own summary are Self-Refine: the model looks at what it wrote. Asking each year on its own, in a new chat, is Chain-of-Verification. And answering from the manual, with its section, is grounding with a citation.",
    )
    return


@app.cell(hide_code=True)
def text_cite():
    mo.md(r"""
    ### Exercise 2. Which answer can you check?
    """)
    return


@app.cell(hide_code=True)
def choice_cite_cell():
    choice_cite = L.choice(
        "cite",
        question="A model got the opening hours of a library as four numbered sentences, and the question *When does the library close on Saturday?* Which answer can you check fastest?",
        options={
            "sure": "*It closes at 14:00. I am sure of that.*",
            "cited": "*At 14:00 [2].*",
            "usual": "*Libraries often close early on Saturday, around 14:00.*",
        },
        correct="cited",
        why={
            "sure": "*I am sure* is more text the model wrote, not a place you can look. Look at the new word *citation* again.",
            "cited": "[2] says where to look: you read sentence 2, and see whether it says 14:00.",
            "usual": "This answer says what libraries often do, not what this source says: there is nothing to look up. Look at the new word *grounding* again.",
        },
    )
    choice_cite
    return (choice_cite,)


@app.cell(hide_code=True)
def text_fresh():
    mo.md(r"""
    ### Exercise 3. Why a fresh prompt?
    """)
    return


@app.cell(hide_code=True)
def choice_fresh_cell():
    choice_fresh = L.choice(
        "fresh",
        question="In Chain-of-Verification, why is each check asked in a fresh prompt, without the draft?",
        options={
            "short": "Short prompts cost less",
            "rule": "A model can answer only one question per prompt",
            "agree": "With its draft in front of it, a model tends to agree with what it wrote",
        },
        correct="agree",
        why={
            "short": "Shorter prompts do cost less, but that is not why. What does a model do with its own draft in front of it? Look at the idea of Chain-of-Verification again.",
            "rule": "A model can answer several questions in one prompt. Look at the idea of Chain-of-Verification again: what should the check not see?",
            "agree": "Every token a model writes depends on all text before it. With the draft in the prompt, the draft's answer is the likely one; a fresh question has to be answered on its own.",
        },
    )
    choice_fresh
    return (choice_fresh,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Ask Qwen3 yourself

    The box holds the prompt of experiment 4, with the source. Change the question at the end, or the source, and see what Qwen3 answers, and which sentence it names. The answer is sampled with seed 0. Qwen3 runs on your own computer, so nothing you type leaves it.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1.5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*The chat model Qwen3-0.6B is not on your computer yet. The button downloads it once from Hugging Face, `Qwen/Qwen3-0.6B` at a fixed version, 1.5 GB, under the Apache-2.0 license, with the tokenizer of Qwen3-0.6B-Base.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 1.5 GB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    mo.md("*Qwen3-0.6B is downloaded.*")
    return


@app.cell(hide_code=True)
def prompt_box_cell():
    prompt_box = mo.ui.text_area(value=C.grounded(C.ASK_SATURDAY)[0]["content"], label="Your prompt", rows=9, full_width=True)
    ask_button = mo.ui.run_button(label="Ask Qwen3")
    mo.vstack([prompt_box, ask_button])
    return ask_button, prompt_box


@app.cell(hide_code=True)
def demo_answer(ask_button, prompt_box):
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif not ask_button.value:
        _out = mo.md("*Press the button to ask Qwen3.*")
    elif not prompt_box.value.strip():
        _out = mo.md("*Write a prompt in the box first.*")
    else:
        _words = {"title": "What Qwen3 answers", "loading": "Loading Qwen3 …", "writing": "Qwen3 is writing …",
                  "done": "{tokens} tokens in {seconds} seconds; the sentences it names: {cited}", "nothing": "none",
                  "load": "Loading Qwen3 stops with an error. Close other programs and press the button again; if it happens again, report it on GitHub.",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(C.ask_live(prompt_box.value, lambda _html: mo.output.replace(mo.Html(_html)), _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en), the form of one prompt changed the answer; here, prompts in a row give you places to check the answers. Later in the course, in *What an agent is: a model, tools and a loop*, a program asks a model again and again, as in a chain, but the model chooses the next step itself, and calls tools. In the domain *Building with AI*, grounding comes back: a program first finds the right text, and then the model answers from it.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_cite, choice_fresh, result_habits):
    L.wrap_up(result_habits, choice_cite, choice_fresh)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
