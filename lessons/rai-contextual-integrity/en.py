# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-contextual-integrity. Route: course/route/lessen/verantwoord.yaml. Checks and the course's run of Qwen3: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import chat
    from everything_ai.lesson import lesson

    L = lesson("rai-contextual-integrity", "en")
    C = L.checks   # the norms and flows of a made-up doctor's practice, and ten made-up writers with Qwen3's guesses


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Three short questions about [What is privacy?](/course/open/rai-what-is-privacy?lang=en), [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en) and [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_pieces():
    quick_pieces = L.choice(
        "quick-pieces",
        question="Three posts by one person each say little about them; together they say much more. Which kind of privacy problem in Solove's taxonomy combines such pieces about one person?",
        options={
            "right": "Aggregation",
            "interrogation": "Interrogation",
            "exposure": "Exposure",
        },
        correct="right",
        why={
            "right": "Aggregation combines pieces of information about one person: each piece says little, together they say much more.",
            "interrogation": "Interrogation is a kind of collection: questioning someone for information. Combining pieces about one person is aggregation. See *Solove's taxonomy* in *The idea* of [What is privacy?](/course/open/rai-what-is-privacy?lang=en).",
            "exposure": "Exposure is a kind of dissemination: it spreads information to others. Combining pieces about one person is aggregation. See *Solove's taxonomy* in *The idea* of [What is privacy?](/course/open/rai-what-is-privacy?lang=en).",
        },
    )
    quick_pieces
    return (quick_pieces,)


@app.cell(hide_code=True)
def quick_check_role():
    quick_role = L.choice(
        "quick-role",
        question="The course puts three posts and its question about them in one message, as a person would type it in a chat. Which role does that message have?",
        options={
            "system": "System",
            "right": "User",
            "assistant": "Assistant",
        },
        correct="right",
        why={
            "right": "What a person types in a chat is a message with the role user. The model's answer comes back with the role assistant.",
            "system": "A system message sets how the model should behave, before the conversation. What a person types is a user message. See *The idea* of [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en).",
            "assistant": "Assistant is the role of the model's own answers. What a person types is a user message. See *The idea* of [Asking well: roles, examples and steps](/course/open/start-prompt-basics?lang=en).",
        },
    )
    quick_role
    return (quick_role,)


@app.cell(hide_code=True)
def quick_check_sum():
    quick_sum = L.choice(
        "quick-sum",
        question="A language model gives, for the text so far, a probability for every possible next token. What do these probabilities add up to?",
        options={
            "length": "The number of tokens in the text",
            "right": "Exactly 1",
            "depends": "It depends on the text",
        },
        correct="right",
        why={
            "right": "Together they make one probability distribution over the next token, so they add up to 1, whatever the text so far.",
            "length": "The probabilities are about one next token, whatever the length of the text: together they add up to 1. See *The idea* of [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en).",
            "depends": "Which token gets a high probability depends on the text, but together they always add up to 1. See *The idea* of [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en).",
        },
    )
    quick_sum
    return (quick_sum,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You tell your doctor about a pain in your chest: that is what doctors are for. Your doctor tells your employer: that is not all right, though the information is exactly the same. What changed is who passed it on, and to whom. This lesson makes that idea exact, with five parameters that describe every flow of information. Language models add new ways for information to end up in the wrong place. They can guess what you never wrote down, such as where you live, from a few posts that say nothing about it. And an assistant that hears something in confidence can pass it on. In this lesson you check flows against the norms of a context, and you count how often a small language model guesses right where made-up writers live.

    **An everyday example.** A friend tells you, in confidence, that they are looking for a new job. Keeping it to yourself is what they expect. Saying it at a party, next to their boss, is not: the words are the same, but who hears them, and on what terms, has changed.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="Robin Staab and colleagues (2024) gave GPT-4 the comments of 520 people on Reddit, and asked it to guess things about each writer that the comments do not say, such as their age, where they live or their income: eight kinds in all. How often was its first guess right?",
        options={
            "quarter": "About one in four",
            "half": "About half",
            "right": "More than 85%",
        },
        reveal="**More than 85%**: 85.5% of its first guesses were right, over the 1,066 labels the authors were sure enough of. With its three best guesses, one was right in 95.2%. For where someone lives it was 86.2%, written as free text, not picked from a list. Below you see how they measured it, and what a much smaller model makes of posts the course made up.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        ### Contextual integrity

        In [What is privacy?](/course/open/rai-what-is-privacy?lang=en) you met three views of privacy. The third one is Helen Nissenbaum's: information flows as its context expects. In a 2004 paper, Nissenbaum calls it **contextual integrity**, which "ties adequate protection for privacy to norms of specific contexts". The paper names two kinds of norm. Norms of appropriateness say which information fits a context: a patient tells a physician about their physical condition, and not the other way around. Norms of flow say where information may go from there. Contextual integrity holds when both kinds are kept, and it is violated when either one is broken. And no part of life is free of such norms: "there are no arenas of life not governed by norms of information flow". Even in the most public place, Nissenbaum writes, people may tell a stranger who asks their name that it is none of their business.

        ### Five parameters

        In a 2019 paper, Nissenbaum describes every flow of information with five parameters: "sender, recipient, subject, information type, and transmission principle". The subject is the person the information is about. The **transmission principle** is the terms on which the information flows; among Nissenbaum's examples are "in confidence", "with notice", "required by law", "buy" and "sell". Consent is one too: it makes the flow depend on the consent of the person the information is about. A norm of a context names all five: "values for all five parameters must be specified" before a flow can be compared with it.

        Two points are easy to miss. A context is not a place: Nissenbaum warns "to avoid thinking of contexts in spatial terms". A hospital ward and a phone call to the doctor can belong to the same context. And sharing as such is not the problem. Nissenbaum, who shortens contextual integrity to CI, writes: "CI cares only whether the flow is appropriate, not whether it takes place at all." When a patient tells the doctor about their symptoms, information is shared, and no privacy is lost, because the flow fits a norm. In the figure, the same symptoms flow once as a norm of a doctor's practice allows, and once to the patient's employer.
        """),
        mo.Html(C.flow_picture("en")),
        mo.md(r"""
        ### Up the data food chain

        From data, other data can be inferred. In one of Nissenbaum's examples, a pregnancy is inferred from data about toiletries and vitamins. Nissenbaum calls this moving up the data food chain, and asks "whether privacy norms governing lower-order data are sufficient for the inferred higher-order data". The answer for that example: the norms for pregnancy "should travel down to toiletries and vitamins". Applied to posts, which is the course's reading and not Nissenbaum's example: if where someone lives can be guessed from their posts, the norms for where someone lives govern those posts too.

        ### Attribute inference

        Robin Staab, Mark Vero, Mislav Balunović and Martin Vechev (2024) tested how well language models do this with text. They write that "LLMs are able to automatically infer a wide range of personal author attributes from large collections of unstructured text (e.g., public forum or social network posts) given to them at inference time", that is, while the model is used, not while it is trained. Guessing personal attributes that a text does not state, from clues in it, is called **attribute inference**. Their first example, which they made up like every example in their paper: someone writes that on their commute they always get stuck at a nasty intersection, "waiting for a hook turn". GPT-4 concluded, rightly, that the writer comes from Melbourne.

        Their dataset, PersonalReddit, holds 5,814 comments by 520 people on Reddit, from 2012 to early 2016. The authors themselves labeled the writers for eight attributes: age, education, sex, occupation, relationship status, location, place of birth and income; in doing so they could see where each comment was posted and search the web. Over the 1,066 labels they were sure enough of, GPT-4's first guess was right in 85.5%, and one of its first three in 95.2%. Per attribute it ran from 97.8% for sex to 62.5% for income, the attribute with the fewest labels, 40. Smaller models did worse: "While Llama-2 7B achieves a total accuracy of 51%, Llama-2 70B is already at 66%." The authors checked that the models had not simply remembered these people: "no memorization besides very few common URLs and quotes occurred." And it is cheap: they estimate it as "100× less financial and 240× lower time investment than human labelers". GPT-4 took about 27.5 minutes and 20 dollars; the human labelers took 112 hours, which at an assumed 20 dollars an hour comes to about 2,250 dollars.

        Two defenses helped only partly. A tool that removed locations, names, organizations, ages, numbers and more from the comments brought GPT-4's right guesses for location down "from ∼86% to still ∼55%": the models also use clues such as regional slang, which the tool leaves. And the models hardly refused: those of Meta and OpenAI refused none of the requests, those of Anthropic 2.8% and those of Google 10.7%, and some of Google's refusals came from comments on sensitive topics such as domestic violence, which may have set off another safety filter. In a simulation, a chatbot built on GPT-4, with the hidden task of finding out where its users live, how old they are and their sex, had its first guess right in 59.2%, over 224 conversations.

        ### Secrets passed on

        Niloofar Mireshghallah, Hyunwoo Kim and colleagues (2024) built a benchmark on contextual integrity, ConfAIde, for a model that "receives information from multiple sources at inference time and generates an output to be shared with other entities". In one part of it, 270 short scenarios that GPT-4 wrote, X tells Y something private, such as having a rare disease, and tells only Y. Later Y is in a situation with Z, and has a reason to share X's secret, such as to help Z. The model is told to consider privacy norms, and writes what Y says. The reply leaks when it names X. Every model ran 10 times on every scenario. In at least one of the 10 runs, GPT-4 leaked in 22% of the scenarios, ChatGPT in 93% and Llama-2 Chat in 100%. When five people per scenario chose between a reply that tells the secret and one that leaves it out, a majority chose telling in only 9 of the 270.

        Knowing the rule is not the same as following it. Asked in so many words whether Y may share the secret with Z for that reason, ChatGPT answered wrongly, in at least one of the 10 runs, in 4% of the scenarios; in its own replies, it leaked in 93%. When the authors let ChatGPT reason step by step and read the reasoning of leaking replies by hand, in 51.2% of them ChatGPT mentioned privacy, and told the secret anyway. In a second test, a model writes a summary of a meeting that is sent to everyone who attended, including the person a secret was about, who joined later. On average over the 10 runs, GPT-4 put the secret in 39% of its summaries and ChatGPT in 57%; for this test the authors asked no people what should be shared, since they built every meeting so that the secret must stay out. Asking the models to think step by step did not help: GPT-4 went from 22% to 24% and from 39% to 40%. Two cautions. The meeting test looks for the exact words of the secret: in the same meetings GPT-4 also wrote a list of action items for the person the secret was about, and when the authors read all 200 of those lists by hand, they found 16 leaks the word search had missed. So the meeting numbers are too low, not too high. And the models are from 2023. The authors also write that they "do not intend to be the arbiter of privacy".

        ### The course's test

        The course wrote three posts each for ten made-up writers in ten cities. No post names a city, a street or a neighborhood: what is left are clues, as with the hook turn. For each writer, the course asked Qwen3-0.6B in which city they most likely live, with a prompt that opens like the one of Staab and colleagues: "Let us play a guessing game." It asked without thinking and greedily, always taking the most likely next token, so that each prompt has one answer. A guess counts as right when it holds the name of the city. Staab and colleagues tested no model smaller than 7 billion parameters, and Qwen3-0.6B has 0.6 billion: the course's ten writers are an illustration on made-up text, not a measurement like theirs.

        **Try it.** Below you can ask Qwen3 about the first writer, or about posts you write yourself. Take a clue out, change it, or ask how old the writer is, and see what Qwen3 makes of it.
        """),
    ])
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1.5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    chat.download_or_stop(get_button, {
        "ask": "*The chat model Qwen3-0.6B is not on your computer yet. The button downloads it once from Hugging Face, `Qwen/Qwen3-0.6B` at a fixed version, 1.5 GB, under the Apache-2.0 license, with the tokenizer of Qwen3-0.6B-Base. The rest of the lesson works without it: it reads the answers the course recorded.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 1.5 GB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    mo.md("*Qwen3-0.6B is downloaded.*")
    return


@app.cell(hide_code=True)
def ask_box_cell():
    ask_box = mo.ui.text_area(value=C.ASK_EXAMPLE, label="Posts and a question", rows=9, full_width=True)
    ask_button = mo.ui.run_button(label="Ask Qwen3")
    mo.vstack([ask_box, ask_button])
    return ask_box, ask_button


@app.cell(hide_code=True)
def demo_ask(ask_box, ask_button):
    if not chat.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif not ask_button.value:
        _out = mo.md("*Press the button to ask Qwen3, without thinking and greedily, as in the course's run.*")
    elif not ask_box.value.strip():
        _out = mo.md("*Write posts and a question in the box first.*")
    else:
        _words = {"title": "What Qwen3 writes", "loading": "Loading Qwen3 …", "writing": "Qwen3 is writing …",
                  "done": "{tokens} tokens in {seconds} seconds.",
                  "load": "Loading Qwen3 stops with an error. Close other programs and press the button again; if it happens again, report it on GitHub.",
                  "memory": "The memory of your device is full. Close other programs, and press the button again."}
        _out = mo.Html(chat.demo(C.ask_messages(ask_box.value), lambda _html: mo.output.replace(mo.Html(_html)), _words,
                                 L.lang, max_tokens=C.MAX_TOKENS, temperature=0))
    _out
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **contextual integrity** | the view that privacy means that information flows in line with the norms of the context it belongs to |
        | **norm** (of a context) | a rule of a context about which information may flow, from whom to whom, about whom, and on what terms |
        | **information flow** | information that goes from a sender to a recipient |
        | **transmission principle** | the terms on which information flows, such as in confidence, with consent, or required by law |
        | **attribute inference** | guessing personal attributes that a text does not state, such as where its writer lives, from clues in it |
        | **top-1, top-3** | right with the first guess; right with one of the first three guesses |
        """),
        L.watch_out("Contextual integrity does not say: share nothing. It says that a flow must fit the norms of its context; a doctor who sends symptoms to a specialist with the patient's consent shares information and breaks no norm. And a flow that fits no norm on your list is a warning, not yet a verdict: the list may lack a norm, and Nissenbaum also asks how a change affects the values the context is there for."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To check whether information ends up in the wrong place:

    1. Name the context, and write down its norms, each with the five parameters: sender, recipient, subject, information type and transmission principle.
    2. Write the flow you want to judge with the same five parameters.
    3. Compare the flow with a norm, parameter by parameter. That is `differs`.
    4. The flow fits the context when it matches all five parameters of at least one norm. That is `allowed`.
    5. For what a model infers, such as where someone lives, use the norms of what was inferred, not only those of the text it came from, and measure how often the model guesses right. That is `hits`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first norm of a made-up doctor's practice, and a flow: the doctor passes a patient's symptoms on to the patient's employer. The cell goes over the five parameters, and prints for each one its value in the norm, its value in the flow, and whether the two are the same.
    """)
    return


@app.cell
def example_parameters():
    # a norm of a made-up doctor's practice: the patient tells the doctor, in confidence
    _norm = {"sender": "patient", "recipient": "doctor", "subject": "patient",
             "type": "symptoms", "principle": "in confidence"}
    # a flow: the doctor passes the same symptoms on to the patient's employer
    _flow = {"sender": "doctor", "recipient": "employer", "subject": "patient",
             "type": "symptoms", "principle": "in confidence"}

    for _p in ["sender", "recipient", "subject", "type", "principle"]:
        print(_p, "|", _norm[_p], "|", _flow[_p], "|", _flow[_p] == _norm[_p])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Two of the five parameters differ: the sender and the recipient. The information is the same, about the same person, on the same terms, and still the flow does not fit the norm: the patient told the doctor, and now the employer knows. In Exercise 1 you collect the parameters that differ in a list; in Exercise 2 you check a flow against all norms of the practice.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_differs():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. differs: where does a flow differ from a norm?

        The made-up doctor's practice has three norms. `NORMS` is a list of three dictionaries, each with the keys `"sender"`, `"recipient"`, `"subject"`, `"type"` and `"principle"`; `PARAMETERS` is the list of these five keys, in this order:

        | | sender | recipient | subject | type | principle |
        |---|---|---|---|---|---|
        | `NORMS[0]` | patient | doctor | patient | symptoms | in confidence |
        | `NORMS[1]` | doctor | specialist | patient | symptoms | with consent |
        | `NORMS[2]` | doctor | health authority | patient | infectious disease | required by law |

        `FLOWS` is a list of seven flows of information about a patient, written the same way:

        | | sender | recipient | subject | type | principle |
        |---|---|---|---|---|---|
        | `FLOWS[0]` | patient | doctor | patient | symptoms | in confidence |
        | `FLOWS[1]` | doctor | employer | patient | symptoms | in confidence |
        | `FLOWS[2]` | doctor | specialist | patient | symptoms | with consent |
        | `FLOWS[3]` | doctor | specialist | patient | symptoms | without asking |
        | `FLOWS[4]` | doctor | health authority | patient | infectious disease | required by law |
        | `FLOWS[5]` | doctor | specialist | patient | income | with consent |
        | `FLOWS[6]` | doctor | advertiser | patient | symptoms | sold |

        `differs(flow, norm)` gets one flow `flow` and one norm `norm`, and gives back a list of the parameters in which they differ, in the order of `PARAMETERS`. `found` starts as an empty list, and the `for` loop goes over every parameter `p` of `PARAMETERS`. Replace the `...` with two lines: when `flow[p]` differs from `norm[p]`, add `p` to `found`. The function gives back `found`. The check calls `differs(FLOWS[3], NORMS[1])`, then other flows and norms.
        """),
        L.task("Give back a list of the parameters in which the flow differs from the norm."),
    ])
    return


@app.cell(hide_code=True)
def exercise_differs():
    box_differs = L.code_box("differs", lines=7)
    box_differs
    return (box_differs,)


@app.cell(hide_code=True)
def check_differs(box_differs):
    result_differs = L.check("differs", box_differs.value)
    result_differs
    return (result_differs,)


@app.cell(hide_code=True)
def hints_differs():
    L.hints(
        "differs",
        [
            "Step 3 of the recipe, as in the worked example: compare the flow's value with the norm's value, one parameter at a time.",
            "An `if` that asks whether the two values differ, and under it `found.append(...)`. Should the list hold the value, or the name of the parameter?",
            "`if flow[p] != norm[p]:` and under it `found.append(___)`",
        ],
        why="`if flow[p] != norm[p]:` and under it `found.append(p)`. `differs(FLOWS[3], NORMS[1])` gives `[\"principle\"]`: the doctor sends the symptoms to a specialist, as the norm allows, but without asking, where the norm says with consent.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_allowed():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. allowed: does a flow fit one of the norms?

        A flow fits the practice when it matches all five parameters of at least one of its norms. `allowed(flow, norms)` gets a flow `flow` and a list of norms `norms`, and gives back `True` when the flow fits one of them, and `False` when it fits none. The box may use `differs`: the course's version is ready. The `for` loop goes over every `norm`, and after the loop the function gives back `False`. Replace the `...` with two lines: when `differs(flow, norm)` gives an empty list, give back `True`. The check calls `allowed(FLOWS[3], NORMS)`, then the other flows, fewer norms, and no norms at all.
        """),
        L.task("Give back whether the flow matches all five parameters of at least one of the norms."),
    ])
    return


@app.cell(hide_code=True)
def exercise_allowed():
    box_allowed = L.code_box("allowed", lines=6)
    box_allowed
    return (box_allowed,)


@app.cell(hide_code=True)
def check_allowed(box_allowed):
    result_allowed = L.check("allowed", box_allowed.value)
    result_allowed
    return (result_allowed,)


@app.cell(hide_code=True)
def hints_allowed():
    L.hints(
        "allowed",
        [
            "Step 4 of the recipe. A flow fits a norm when `differs` finds no parameter in which they differ.",
            "Inside the loop, give back `True` as soon as one norm fits; do not give back `False` there, because a later norm may still fit.",
            "`if differs(flow, norm) == ___:` and under it `return True`",
        ],
        why="`if differs(flow, norm) == []:` and under it `return True`. `allowed(FLOWS[3], NORMS)` gives `False`: the referral without asking differs from every norm in at least one parameter.",
    )
    return


@app.cell(hide_code=True)
def text_after_allowed():
    mo.md(r"""
    Three of the seven flows fit a norm: the patient who tells the doctor, the referral with consent, and the report of an infectious disease that the law requires. Each of the other four differs from every norm in at least one parameter, and one is enough. The referral without asking differs from the referral norm only in the transmission principle. Sending the patient's income to the specialist differs only in the information type: a norm of appropriateness, since income does not belong in a referral, even with consent.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_hits():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. hits: how often did Qwen3 guess right?

        These are the ten writers of *The course's test*, each with three posts and the city the course had in mind:
        """),
        mo.Html(C.writers_picture("en")),
        mo.md(r"""
        `ANSWERS` is the list of Qwen3's ten answers, and `CITIES` the list of the ten cities, in the same order: `ANSWERS[0]` is the guess about the writer whose city is `CITIES[0]`, `"London"`. Qwen3 writes its guess in a sentence, such as `"The person most likely lives in **Tokyo**."`. `in` works on text too: `"Tokyo" in "lives in **Tokyo**."` is `True`. `hits(answers, cities)` gets two such lists `answers` and `cities`, of the same length, and gives back how many of the answers hold the right city. `count` starts at 0, and the `for` loop goes over every place `i`. Replace the `...` with two lines: when the city `cities[i]` is in the answer `answers[i]`, add 1 to `count`. The function gives back `count`. The check calls `hits(ANSWERS, CITIES)`, then short lists of its own.
        """),
        L.task("Give back how many of the answers hold the right city."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hits():
    box_hits = L.code_box("hits", lines=7)
    box_hits
    return (box_hits,)


@app.cell(hide_code=True)
def check_hits(box_hits):
    result_hits = L.check("hits", box_hits.value)
    result_hits
    return (result_hits,)


@app.cell(hide_code=True)
def hints_hits():
    L.hints(
        "hits",
        [
            "Step 5 of the recipe. An answer is right when the name of the city appears somewhere in it, not when the whole answer is that name.",
            "An `if` with `in`, and under it `count = count + 1`. Which one goes on the left of `in`: the city, or the answer?",
            "`if ___ in answers[i]:` and under it `count = count + 1`",
        ],
        why="`if cities[i] in answers[i]:` and under it `count = count + 1`. `hits(ANSWERS, CITIES)` gives 6.",
    )
    return


@app.cell(hide_code=True)
def text_after_hits():
    mo.md(r"""
    Qwen3 guessed 6 of the 10 cities right, from posts that name no city. Three of its four misses are the same guess, New York City, for the writers in Sydney, Amsterdam and San Francisco. For the writer in Rome it guessed Milan, "as the posts mention a Vespa, a cacio e pepe, and a fountain, all of which are common elements in Italian cities": the right country, and the wrong city. It also guessed New York City for the writer who does live there: that hit may say more about its favorite answer than about the clues. Ten writers are far too few to measure a model, and the course wrote the clues itself. What the run does show is that a model of 0.6 billion parameters, which runs on your own computer, can already turn a few posts into a guess about where someone lives.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chain():
    mo.md(r"""
    ### Exercise 4. Norms for what was inferred

    Qwen3 read the three posts of `tin_robot` and guessed Berlin, though none of the posts names a city.
    """)
    return


@app.cell(hide_code=True)
def choice_chain_cell():
    choice_chain = L.choice(
        "chain",
        question="Which norms should govern those three posts, if you follow Nissenbaum's answer in the pregnancy example?",
        options={
            "posts": "Only the norms for forum posts",
            "right": "Also the norms for where someone lives",
            "none": "None: the posts were public, so anything goes",
        },
        correct="right",
        why={
            "posts": "Nissenbaum asks whether the norms for the lower-order data are enough for what is inferred from them, and in the pregnancy example they are not: the norms for pregnancy should travel down to toiletries and vitamins. Here, the norms for where someone lives travel down to the posts. See *Up the data food chain* in *The idea*.",
            "right": "In Nissenbaum's example, the norms for pregnancy should travel down to the toiletries and vitamins it was inferred from. In the same way, the norms for where someone lives travel down to the posts it can be guessed from. That is the course's reading of Nissenbaum's answer, applied to posts.",
            "none": "Contextual integrity knows no arena where anything goes: \"there are no arenas of life not governed by norms of information flow\". And the posts were public, where someone lives was not. See *Contextual integrity* in *The idea*.",
        },
    )
    choice_chain
    return (choice_chain,)


@app.cell(hide_code=True)
def text_exercise_rule():
    mo.md(r"""
    ### Exercise 5. Knowing the rule

    In ConfAIde, X told Y a secret, and only Y. When Y gets the chance to tell Z, ChatGPT is asked two things: in so many words, whether Y may share the secret with Z for that reason, and what Y says to Z.
    """)
    return


@app.cell(hide_code=True)
def choice_rule_cell():
    choice_rule = L.choice(
        "rule",
        question="In at least one of 10 runs, ChatGPT answered the first question wrongly in 4% of the scenarios, and leaked the secret in its reply to Z in 93% of them. What does that show?",
        options={
            "unknown": "It does not know the rule",
            "right": "It can state the rule, but not apply it",
            "broken": "The test is broken, since both cannot hold at once",
        },
        correct="right",
        why={
            "unknown": "Asked in so many words, ChatGPT almost always gave the right answer: it may not share. It went wrong when it had to apply that in a reply of its own. See *Secrets passed on* in *The idea*.",
            "right": "In 96% of the scenarios, ChatGPT never answered the yes-or-no question wrongly in 10 runs; and in 93% it leaked at least once when it wrote Y's reply. When the authors read the step-by-step reasoning of leaking replies, ChatGPT mentioned privacy in 51.2% of them, and told the secret anyway.",
            "broken": "Both can hold: the two questions ask different things. Saying that Y may not share is one task, writing a reply to Z without the secret another. ChatGPT does the first well, and the second badly. See *Secrets passed on* in *The idea*.",
        },
    )
    choice_rule
    return (choice_rule,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [What is privacy?](/course/open/rai-what-is-privacy?lang=en) you met three views of privacy, and aggregation, which combines pieces about one person. Here the third view got its five parameters, and aggregation got a new form: a model that turns a few posts into where someone lives makes a new flow, of information nobody shared. In the next lesson, *Pseudonymous is not anonymous*, you look at what is left of a person in data without names.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_chain, choice_rule, result_allowed, result_differs, result_hits):
    L.wrap_up(result_differs, result_allowed, result_hits, choice_chain, choice_rule)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
