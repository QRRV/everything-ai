# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-ai-at-work. Route: course/route/lessen/bouwen.yaml. Checks and the course's run of Qwen3: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import chat
    from everything_ai.lesson import lesson

    L = lesson("build-ai-at-work", "en")
    C = L.checks   # the ten tasks, how an answer is marked, and the course's run of Qwen3


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Prompts that check themselves](/course/open/start-prompt-chains?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_steps():
    quick_steps = L.choice(
        "quick-steps",
        question="Qwen3 checked numbers with a plain yes or no, and said yes to all 30 of them, the wrong ones too. Which check caught more of the wrong ones?",
        options={
            "twice": "The same yes-or-no question, asked twice",
            "steps": "A check that works the answer out first, with steps you can read",
            "none": "None: a model cannot check anything",
        },
        correct="steps",
        why={
            "steps": "Working the answer out gave the model room, and you can read where it went wrong. It still missed some, so you read its steps, not just its last line.",
            "twice": "A model that says yes once will most likely say yes again. What gave the model more room? See [Prompts that check themselves](/course/open/start-prompt-chains?lang=en).",
            "none": "One kind of check did catch some of the wrong numbers. Which one? See [Prompts that check themselves](/course/open/start-prompt-chains?lang=en).",
        },
    )
    quick_steps
    return (quick_steps,)


@app.cell(hide_code=True)
def quick_check_source():
    quick_source = L.choice(
        "quick-source",
        question="You give an assistant the opening hours of a shop as text, and ask when it closes on Saturday. Why ask it to name the sentence it used?",
        options={
            "faster": "Then the model answers faster.",
            "always": "Then the answer is always right.",
            "check": "Then you can read that sentence and see whether the answer is in it.",
        },
        correct="check",
        why={
            "check": "A citation points you to the sentence, so checking takes seconds. And the citation itself can be wrong, so you do read it.",
            "faster": "A citation makes the answer a little longer, not faster. What does the sentence number give you? See [Prompts that check themselves](/course/open/start-prompt-chains?lang=en).",
            "always": "Qwen3 sometimes named the wrong sentence, even with the source in front of it. What is a citation for, then? See [Prompts that check themselves](/course/open/start-prompt-chains?lang=en).",
        },
    )
    quick_source
    return (quick_source,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    At work, AI can save you hours on one task and cost you a mistake on the next, and the two tasks can look just as hard. Knowing which is which, and how you work together with AI on each, decides whether it makes your work better or worse.

    **An everyday example.** A navigation app knows the streets of a whole country, and gets you through a strange city without a wrong turn. But now and then it sends you down a road that has been closed for weeks, and it sounds just as sure as always. Drivers who use it a lot learn where to trust it, and where to look out of the window themselves. That is only a comparison; below you see the same thing for AI at work.
    """)
    return


@app.cell(hide_code=True)
def predict_outside():
    guess_outside = L.guess(
        "outside-frontier",
        question="In 2023, researchers gave 758 consultants of Boston Consulting Group realistic tasks, with GPT-4 or without it. One task was picked because GPT-4 would get it wrong. On that task, how often did consultants with GPT-4 find the right answer, compared with consultants without it?",
        options={
            "more": "More often: with AI, people work better",
            "same": "About as often: the AI made no difference",
            "less": "Less often",
        },
        reveal="**Less often.** Without AI, 84.5% of the consultants found the right answer; with GPT-4, 60% and 71% in the two groups that had it: on average 19 percentage points fewer. Many took over the model's convincing but wrong analysis, and checked it less. On 18 other tasks, which the model could do well, the same consultants with GPT-4 finished 12.2% more tasks, 25.1% faster, and their work was rated more than 40% better. (Dell'Acqua and others, *Navigating the Jagged Technological Frontier*, 2023.)",
    )
    guess_outside
    return (guess_outside,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **jagged frontier** | the uneven border between tasks AI does well and tasks it gets wrong; two tasks that look just as hard to a person can lie on different sides of it |
    | **centaur** | a way of working with AI in which you split the work: some parts go to the AI as a whole, the rest you do yourself, and you check what comes back |
    | **cyborg** | a way of working with AI in which you and the AI go back and forth at every step, sentence by sentence, so closely that you can hardly tell whose part is whose |

    The study of the consultants used all three words. The name *centaur* comes from chess, where a person and a chess program play together as one team, like the centaur of the old Greek stories: half human, half horse.

    ### The frontier is jagged

    The border is not a straight line, with easy tasks inside and hard ones outside. It goes in and out: a model can do things that are hard for you, and fail at things a child can do. You cannot tell from the task; you find out by trying it, on cases where you know the answer.

    The course gave Qwen3-0.6B, the small chat model of [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en), ten small tasks, each eight times. Each answer was marked right or wrong by a rule the course wrote for that task.
    """)
    return


@app.cell(hide_code=True)
def picture_frontier():
    mo.vstack([
        mo.Html(C.frontier(L.lang)),
        mo.md(C.frontier_after(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def picture_tasks():
    L.explore("task-answers", "task", lambda _v: C.task_frame(int(_v), L.lang), start=2, low=1, high=len(C.TASKS), step=1, role="",
              task="Slide through the ten tasks and read what Qwen3 wrote. Which task surprises you most?")
    return


@app.cell(hide_code=True)
def text_moves():
    mo.vstack([
        L.watch_out("The frontier of one model is not that of another, and it moves with every new version. A large assistant names the capital of Australia without trouble, and still has tasks it gets wrong that look just as easy. So you test the model you actually use."),
        mo.md(r"""
        Why does a model that multiplies 23 by 47 not see three r's in *strawberry*? Part of the answer: a model reads text in tokens, pieces of words, not letter by letter, so counting the letters of a word is hard for it. Later in the course you see how a text is cut into tokens. But you do not need the reason to see the frontier: trying is enough.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: AI in your daily work

    1. **Split the work into tasks.** Not *the newsletter*, but: collect the results, write the opening, check the dates, translate the summary.
    2. **Find out where each task lies.** Try it a few times on a case you know the answer to. Right every time: inside the frontier. Wrong even once, in a way you would not notice: outside.
    3. **Inside: pick your way of working.** As a centaur, you hand over the whole task and check the result as a whole. As a cyborg, you work along with it, sentence by sentence.
    4. **Outside: do it yourself,** or let the AI do a part you check line by line.
    5. **Prompt as you would brief a colleague.** Say who it is for and what it is for, give the text it needs, ask for the form you want, and ask again with what was missing.
    6. **Test again** when you switch to another model, or a new version comes out: the frontier moves.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    You write the monthly newsletter of a sports club, and want to use an assistant for it.

    1. **Split.** Four tasks: the match results of the month, an opening paragraph, the dates of the next matches, and a short summary in English for the members who came from abroad.
    2. **Where does each lie?** You try the results: the assistant makes a neat table of your notes, but once it swaps two scores. That is outside the frontier, in a way you would not notice. The dates of the next matches it cannot know at all. The opening paragraph and the summary come out well every time you try.
    3. **Inside.** The opening you write as a cyborg: you ask for three versions, take a sentence from one, ask again with a joke from the board meeting. The summary you do as a centaur: you hand over the finished newsletter, and read the English once through.
    4. **Outside.** The results and the dates you type yourself, from the league's website. Or the assistant makes the table, and you check every score against the website.
    5. **Prompt.** *This is the newsletter of our football club, for members and parents. Write a summary in English of at most 100 words, in a friendly tone, with the dates exactly as they are.*
    6. **Test again.** Next month the club's assistant gets a new version: you try the results table again before you trust it.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Centaur or cyborg?

        Six people use an assistant at work. Who splits the work, and who goes back and forth at every step?
        """),
        L.task("Put each of the six in the right group."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_ways = L.sort("centaur-or-cyborg")
    sort_ways
    return (sort_ways,)


@app.cell(hide_code=True)
def check_centaur_or_cyborg(sort_ways):
    result_ways = L.check("centaur-or-cyborg", sort_ways)
    result_ways
    return (result_ways,)


@app.cell(hide_code=True)
def hints_ways():
    L.hints(
        "centaur-or-cyborg",
        [
            "Look at how often the work goes back and forth between the person and the assistant.",
            "Once there and once back, with a check at the end: that is splitting the work. Many short turns: that is working along.",
            "Three of the six hand over a whole part and check it afterwards.",
        ],
        why="Centaur: **Minutes**, **Translation** and **Split**: a whole part goes to the assistant, and comes back to be checked. Cyborg: **Sentence by sentence**, **Sparring** and **Line by line**: many short turns, so that the result is made by both.",
    )
    return


@app.cell(hide_code=True)
def text_frontier_qwen():
    mo.md(r"""
    ### Exercise 2. Inside or outside?
    """)
    return


@app.cell(hide_code=True)
def choice_frontier_qwen_cell():
    choice_frontier_qwen = L.choice(
        "frontier-qwen",
        question="At work, you want Qwen3 to turn sentences into shopping lists for the canteen, and to look up capitals for a quiz. Going by the picture of the ten tasks, what do you do?",
        options={
            "both": "Use it for both, and check now and then",
            "lists": "Use it for the lists, which you read once; look the capitals up somewhere else",
            "neither": "Use it for neither: a model that gets a capital wrong cannot be trusted with anything",
        },
        correct="lists",
        why={
            "lists": "The lists were right in all eight runs: inside the frontier, so a quick read is enough. The capital was wrong in all eight, and it sounded sure: outside.",
            "both": "Look at the capital in the picture: wrong in every run. Checking now and then lets most wrong answers through. Which of the two tasks lies outside the frontier?",
            "neither": "The frontier is jagged: a model that fails one task can do another well. Look at the shopping list in the picture.",
        },
    )
    choice_frontier_qwen
    return (choice_frontier_qwen,)


@app.cell(hide_code=True)
def text_outside_study():
    mo.md(r"""
    ### Exercise 3. The task outside the frontier
    """)
    return


@app.cell(hide_code=True)
def choice_outside_study_cell():
    choice_outside_study = L.choice(
        "outside-study",
        question="In the study, the consultants who had GPT-4 found the right answer less often on the task outside the frontier. What do the researchers say went wrong?",
        options={
            "slow": "GPT-4 was so slow that they ran out of time.",
            "trusted": "They relied on the model's analysis, which was convincing but wrong, and questioned it less.",
            "refused": "GPT-4 refused to help with that task.",
        },
        correct="trusted",
        why={
            "trusted": "The data looked complete, but the interview notes held details that pointed the other way, and the model missed them. Its notes read well, so many took them over. Their written advice was even rated better, also when it was wrong.",
            "slow": "Those with GPT-4 were faster on that task, not slower. Look at the reveal under the question about the consultants again.",
            "refused": "GPT-4 helped and gave an answer, a convincing one. Look at the reveal under the question about the consultants again.",
        },
    )
    choice_outside_study
    return (choice_outside_study,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Find the frontier yourself

    Think of a small task from your own work or school, where you know the right answer. Ask Qwen3, and press the button a few times with another seed to see whether it is right every time. Qwen3 runs on your own computer, so nothing you type leaves it.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1.5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    chat.download_or_stop(get_button, {
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
    prompt_box = mo.ui.text_area(value=C.TASKS["letters"][0], label="Your task", rows=3, full_width=True)
    seed_box = mo.ui.number(start=0, stop=99, step=1, value=0, label="Seed")
    ask_button = mo.ui.run_button(label="Ask Qwen3")
    mo.vstack([prompt_box, mo.hstack([seed_box, ask_button], justify="start")])
    return ask_button, prompt_box, seed_box


@app.cell(hide_code=True)
def demo_answer(ask_button, prompt_box, seed_box):
    if not chat.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif not ask_button.value:
        _out = mo.md("*Press the button to ask Qwen3.*")
    elif not prompt_box.value.strip():
        _out = mo.md("*Write a task in the box first.*")
    else:
        _words = {"title": "What Qwen3 answers", "loading": "Loading Qwen3 …", "writing": "Qwen3 is writing …",
                  "done": "{tokens} tokens in {seconds} seconds.",
                  "load": "Loading Qwen3 stops with an error. Close other programs and press the button again; if it happens again, report it on GitHub.",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(chat.demo([{"role": "user", "content": prompt_box.value}], lambda _html: mo.output.replace(mo.Html(_html)), _words, L.lang,
                                 max_tokens=C.MAX_TOKENS, seed=int(seed_box.value)))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Prompts that check themselves](/course/open/start-prompt-chains?lang=en) you let a model check its own answers; here you decide which tasks need that, because they lie outside the frontier, and how you share the work on the rest. The next lessons of this module go through the work itself: searching and learning with AI, making things without code, automating work, and getting help with code.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_frontier_qwen, choice_outside_study, result_ways):
    L.wrap_up(result_ways, choice_frontier_qwen, choice_outside_study)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
