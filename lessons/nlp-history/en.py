# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-history. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-history", "en")
    C = L.checks   # six made-up pairs of Dutch and English sentences, and a made-up shared task with three systems


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Why language is hard for computers](/course/open/nlp-why-language-is-hard?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_readings():
    quick_readings = L.choice(
        "quick-readings",
        question="In *we saw a bat*, *saw* has three meanings and *bat* two. In how many ways can the four words be read?",
        options={
            "plus": "5, since the meanings add up",
            "right": "6, since the meanings multiply",
            "two": "2, since only two words have more than one meaning",
        },
        correct="right",
        why={
            "right": "Each meaning of saw goes with each meaning of bat: 3 × 2 = 6. This lesson shows how NLP went from writing such choices down in rules to learning them from data.",
            "plus": "Each meaning of saw goes with each meaning of bat, so the counts multiply: 3 × 2 = 6. See [Why language is hard for computers](/course/open/nlp-why-language-is-hard?lang=en).",
            "two": "Two words have more than one meaning, and each meaning of one goes with each meaning of the other: 3 × 2 = 6. See [Why language is hard for computers](/course/open/nlp-why-language-is-hard?lang=en).",
        },
    )
    quick_readings
    return (quick_readings,)


@app.cell(hide_code=True)
def quick_check_context():
    quick_context = L.choice(
        "quick-context",
        question="The town council members refused the angry demonstrators a permit because they feared violence. Why does *they* point to the council members?",
        options={
            "grammar": "Grammar: they points to the subject of the sentence",
            "near": "Position: they points to the group that comes closest before it in the sentence",
            "right": "Knowledge of the world: those who fear violence refuse a permit",
        },
        correct="right",
        why={
            "right": "With advocated instead of feared, they points to the demonstrators, and the grammar stays the same. Rules about form cannot see that; this lesson shows what NLP tried next.",
            "grammar": "With advocated instead of feared, they points to the demonstrators, in the same grammar. See [Why language is hard for computers](/course/open/nlp-why-language-is-hard?lang=en).",
            "near": "The closest group before they is the demonstrators, and yet the council members are meant. See [Why language is hard for computers](/course/open/nlp-why-language-is-hard?lang=en).",
        },
    )
    quick_context
    return (quick_context,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Today, an app on your phone translates a menu while you wait. In 1954, a computer translated a few dozen Russian sentences into English with a short list of words and a handful of rules. Between the two lie three ways of building NLP: rules written by hand, statistics counted in large amounts of text, and neural networks. A way of working made each step measurable: shared tasks, in which every system is scored on the same test.

    This lesson walks through that history, lets a program learn a translation by counting, and scores three systems on one test set.

    **An everyday example.** A school exam is the same paper for the whole class, marked with the same key: only then can two grades be compared.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In January 1954, IBM and Georgetown University showed a computer in New York that translated Russian sentences into English. How many words did it know?",
        options={
            "right": "About 250",
            "mid": "About 2,500",
            "many": "About 25,000",
        },
        reveal="**About 250.** John Hutchins (2004) writes of \"a vocabulary of just 250 lexical items (stems and endings) and a limited set of just six rules.\" *Rules* in *The idea* says how it was received.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Rules

    On January 7, 1954, IBM and Georgetown University showed at IBM's headquarters in New York how a computer translated Russian into English, with a vocabulary of 250 stems and endings and six rules. Most of the "more than 60" sentences were about organic chemistry, John Hutchins writes in a history of the event (2004). A newspaper quoted a forecast that "five, perhaps three, years hence, interlingual meaning conversion by electronic process in important functional areas of several languages may well be an accomplished fact". Leon Dostert led the Georgetown team, and "Whether these were Dostert's words is not known", Hutchins adds. The textbook of Jurafsky and Martin, in its draft of August 19, 2026, calls the event "the first public demonstration of an MT system prototype", where MT is short for machine translation. Hutchins (2004) saw it differently: "it had not been a 'prototype' system but a 'showcase' intended to attract attention and funds".

    Twelve years later, in November 1966, a committee called ALPAC concluded, in a report published by the US National Academy of Sciences: "there is no immediate or predictable prospect of useful machine translation." Hutchins (1996): "Its effect was to bring to an end the substantial funding of MT research in the United States for some twenty years."

    Rules did produce striking programs. ELIZA, which Joseph Weizenbaum described in January 1966, answered like a therapist by matching patterns in what people typed and turning them around; "People became deeply emotionally involved with the ELIZA system", Jurafsky and Martin write. Terry Winograd's SHRDLU, described in 1971–72, took instructions about blocks on a table; in Winograd's words (1971): "We can give it instructions to manipulate toy objects". But each program worked in a small world, and each rule had to be written by hand. Jurafsky and Martin: "Symbolic approaches dominated AI, NLP, and cognitive science roughly from 1965 til the early 1990s." Here, symbolic means: built from rules and symbols that people wrote.

    ### Statistics

    The next period, in their words, was "the long revival of empiricism when the field moved back toward statistical and neural models, beginning various times between 1975 and 1988 and lasting until 2017." It started in speech recognition, with Fred Jelinek's group at IBM between 1975 and 1985; according to Jurafsky and Martin, the group "invented the term 'language model'". In 1988, the same group presented a statistical approach to translation at the COLING conference in Budapest, and in 1990 they wrote: "In this paper, we present a statistical approach to machine translation." They learned from the debates of the Canadian parliament, which are published in English and French: "These proceedings are called Hansards". Instead of a dictionary written by hand, they learned from 40,000 pairs of sentences which words translate which. It starts with counting which words occur together in sentences that translate each other.

    Statistics needs text, and text with the right answers in it. The Penn Treebank was described in 1993 as "a corpus consisting of over 4.5 million words of American English", in which people had marked for each word whether it was a noun, a verb or something else. A corpus is a large collection of text, put together for study.

    ### Neural networks

    In 2003, Yoshua Bengio and colleagues described a neural language model in the *Journal of Machine Learning Research* (an earlier version was at the NIPS conference of 2000): "We propose to fight the curse of dimensionality by learning a distributed representation for words". The problem they meant: there are far too many sequences of words to see them all in training. Their answer was to give each word a list of numbers, so that what the model learns from one sentence carries over to sentences with similar words. In 2014, several groups showed that a neural network could translate whole sentences; Jurafsky and Martin write that the approach "was pioneered by Kalchbrenner and Blunsom (2013)". On June 12, 2017, the Transformer appeared, "based solely on attention mechanisms, dispensing with recurrence and convolutions entirely." And on November 30, 2022, OpenAI introduced ChatGPT as a research preview: "We've trained a model called ChatGPT which interacts in a conversational way."
    """)
    return


@app.cell(hide_code=True)
def timeline_cell():
    mo.Html(C.timeline_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Shared tasks

    Each step only counted once others could see that it worked better. A **shared task** is a contest with fixed data and a fixed score: the organizers give every team the same training data and test data, and score every system in the same way. The organizers of CoNLL's shared task of 2018 put it like this: "Every year, the Conference on Computational Natural Language Learning (CoNLL) features a shared task, in which participants train and test their learning systems on the same data sets." The first Workshop on Statistical Machine Translation, WMT, gave every team the same data to train on in 2006, "To have a common framework that allows for comparable results, and also to lower the barrier to entry".

    Some of the best known:

    - **Senseval** held its first exercise in 1998, on choosing the meaning of a word in a sentence. Its fourth edition, in 2007, was called SemEval, because the tasks had grown wider. Its organizers wrote in 2001 that its "main function is not to determine a winner".
    - **CoNLL**'s list of shared tasks starts in 1999. In 2002, the task was finding names of people, places and organizations in Spanish and Dutch; the Dutch data came from "four editions of the Belgian newspaper ‘De Morgen’ of 2000", and twelve systems took part. In 2003, sixteen systems did the same for English and German.
    - **WMT** followed a shared task at a workshop in 2005; since 2016 it has been called the Conference on Machine Translation.

    Mark Liberman, writing about Jelinek's legacy in 2010, called the method "the competitive evaluation of alternative algorithms based on precise quantitative criteria defined in advance, relative to a shared body of training and testing material."
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | rules, statistics, neural networks | three ways of building NLP: knowledge written by hand, counted in text, or learned by a network |
        | parallel text | sentences in one language next to their translation in another, such as the Hansards |
        | **shared task** | a contest with fixed data and a fixed score, so that the systems of all teams can be compared |
        | `C.PAIRS` | six made-up pairs of a Dutch and an English sentence, each as two lists of words |
        | `C.TEST`, `C.ANSWERS`, `C.OUTPUTS` | the ten test sentences of a made-up shared task, the right answers, and the answers of three systems |
        | `for a, b in pairs` | takes each pair in the list apart into its two parts |
        | `x in a_list` | `True` when the list holds `x`: `"kat" in ["de", "kat", "slaapt"]` is `True` |
        """),
        L.watch_out("The periods overlap, and their dates are rough: Jurafsky and Martin write \"roughly\" and \"various times\". Rules never vanished, and counting words is still in use next to neural networks."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To learn a translation by counting, as statistics did:

    1. Take pairs of sentences that translate each other.
    2. Keep the pairs whose sentence in one language holds the word.
    3. Count the words of the other language in those pairs: the most frequent is a candidate translation.

    To score systems, as a shared task does:

    1. Fix the test data and the right answers before the systems run.
    2. Fix the score: here, the share of answers that are right.
    3. Run every system on the same test data, and put the scores side by side.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell goes over the pairs of `C.PAIRS`, takes each pair apart into its Dutch and its English list, and prints the pairs whose Dutch sentence holds *kat*.
    """)
    return


@app.cell
def example_pairs():
    for dutch, english in C.PAIRS:
        if "kat" in dutch:
            print(dutch, english)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Three pairs hold *kat*, and *cat* is in all three; *the* is in two of them. Counting which English word comes with *kat* most often finds its translation without a dictionary.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cooccur():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. cooccur: a translation by counting

        Write `cooccur(word, pairs)`: a dict from each English word to the number of pairs in which it occurs together with the Dutch `word`. Go over the pairs with `for dutch, english in pairs:`; for each pair whose Dutch list holds `word`, go over the English words. Replace the `...`. The check calls `cooccur("kat", PAIRS)`, then the other Dutch words and pairs of its own.
        """),
        L.task("Count the English words in the pairs whose Dutch sentence holds the word."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cooccur():
    box_cooccur = L.code_box("cooccur", lines=10)
    box_cooccur
    return (box_cooccur,)


@app.cell(hide_code=True)
def check_cooccur(box_cooccur):
    result_cooccur = L.check("cooccur", box_cooccur.value)
    result_cooccur
    return (result_cooccur,)


@app.cell(hide_code=True)
def hints_cooccur():
    L.hints(
        "cooccur",
        [
            "The worked example finds the pairs; now count what is in them.",
            "What do you do with a word that is already in `counts`, and with one that is not there yet?",
            "`                    counts[w] = ___`",
        ],
        why="`counts = {}`, `for dutch, english in pairs:`, `if word in dutch:`, `for w in english:`, then `if w in counts:` with `counts[w] = counts[w] + 1`, `else:` with `counts[w] = 1`, and after the loops `return counts`. `cooccur(\"kat\", PAIRS)` gives `{\"the\": 2, \"cat\": 3, \"sleeps\": 1, \"eats\": 1, \"a\": 1, \"plays\": 1}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_cooccur():
    mo.md(r"""
    *Cat* wins for *kat*, and *dog* for *hond*: learned from six pairs, with no dictionary. But for *slaapt*, *the* and *sleeps* tie, because *the* is in most pairs. The IBM models did more than count to see past such words.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: count for other words

    Pick a Dutch word, and see which English words the course's `cooccur` finds next to it.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_word = mo.ui.dropdown(options=C.word_options(L.lang), value=C.DUTCH[0], label="A Dutch word")
    demo_word
    return (demo_word,)


@app.cell(hide_code=True)
def demo_look(demo_word):
    mo.Html(C.demo_picture(demo_word.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_score():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. score: one test for everyone

        `C.TEST` holds the ten test sentences of a made-up shared task: is the marked word a noun or a verb? `C.ANSWERS` holds the right answers, and `C.OUTPUTS` is a dict from each of three systems to its answers. Write `score(predicted, answers)`: the share of places where the list `predicted` has the same answer as the list `answers`, a number between 0 and 1. Count with `for i in range(len(answers)):`, and divide by the number of answers. Replace the `...`. The check calls `score(OUTPUTS["rules"], ANSWERS)`, then the other systems and lists of its own.
        """),
        L.task("Count the places where the two lists agree, and divide by their length."),
    ])
    return


@app.cell(hide_code=True)
def exercise_score():
    box_score = L.code_box("score", lines=6)
    box_score
    return (box_score,)


@app.cell(hide_code=True)
def check_score(box_score):
    result_score = L.check("score", box_score.value)
    result_score
    return (result_score,)


@app.cell(hide_code=True)
def hints_score():
    L.hints(
        "score",
        [
            "*Recipe* says what the score is here.",
            "At which places do you add one to the count, and what do you divide by at the end?",
            "`        if ___:`",
        ],
        why="`right = 0`, `for i in range(len(answers)):`, then `if predicted[i] == answers[i]:` with `right = right + 1` under it, and after the loop `return right / len(answers)`. `score(OUTPUTS[\"rules\"], ANSWERS)` gives `0.7`.",
    )
    return


@app.cell(hide_code=True)
def text_after_score():
    mo.md(r"""
    The system that counts gets 0.8, the one with rules 0.7, and a system that always says noun already gets 0.5. That last number matters: a score only says something next to what a very simple system gets on the same test.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_linguist():
    mo.md(r"""
    ### Exercise 3. A remark about linguists

    A remark attributed to Fred Jelinek is often quoted as "Whenever I fire a linguist our system performance improves". According to Stevenson and Merlo (2022), Jelinek dated it to a talk at a workshop in December 1988; the wording differs from one retelling to the next.
    """)
    return


@app.cell(hide_code=True)
def choice_linguist_cell():
    choice_linguist = L.choice(
        "linguist",
        question="What does the remark stand for in the history of NLP?",
        options={
            "right": "Models learned from data doing better than rules written by experts",
            "wrong": "Linguistics turning out to be wrong about how language works",
            "people": "Faster computers making the knowledge of people unnecessary for language",
        },
        correct="right",
        why={
            "right": "The remark is about systems: models learned from text did better than rules that linguists wrote. It says nothing about whether linguistics is right, and the data was often marked up by linguists, as in the Penn Treebank.",
            "wrong": "It is about systems, not about the science: rules written by hand did worse than models learned from data. See *Statistics* in *The idea*.",
            "people": "The models learned from text that people had written and often marked up, such as the Penn Treebank. What changed is who writes the knowledge down. See *Statistics* in *The idea*.",
        },
    )
    choice_linguist
    return (choice_linguist,)


@app.cell(hide_code=True)
def text_exercise_compare():
    mo.md(r"""
    ### Exercise 4. Two scores

    Two made-up teams report on finding names of people, places and organizations in Dutch news. Team A gets 90 percent right on news it picked itself. Team B gets a score of 80 on the Dutch test set of CoNLL-2002.
    """)
    return


@app.cell(hide_code=True)
def choice_compare_cell():
    choice_compare = L.choice(
        "compare",
        question="Which team has the better system?",
        options={
            "a": "Team A, since 90 is more than 80",
            "right": "You cannot tell, since the two numbers come from different data",
            "b": "Team B, since the test set of a shared task is harder than news picked by a team",
        },
        correct="right",
        why={
            "right": "A score means something only next to scores on the same data with the same measure. That is what a shared task fixes: in the words of WMT, \"a common framework that allows for comparable results\".",
            "a": "Team A chose its own news, which may be easier. Only scores on the same test data can be put side by side. See *Shared tasks* in *The idea*.",
            "b": "Nothing makes the test set of a shared task harder; what it gives is the same data for everyone. See *Shared tasks* in *The idea*.",
        },
    )
    choice_compare
    return (choice_compare,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Why language is hard for computers](/course/open/nlp-why-language-is-hard?lang=en) showed what makes language hard: ambiguity, variation and context. This lesson showed three ways in which NLP tried to cope: rules written by hand, statistics counted in large amounts of text, and neural networks. Shared tasks made each step measurable on the same data. Your `cooccur` learned that *kat* is *cat* from six pairs of sentences, and your `score` compared three systems on one test set. The next lesson, *An NLP pipeline*, shows how classic NLP chains its steps, each one feeding the next.

    **An everyday example.** A recipe that a cook writes down from experience works in that cook's kitchen; a recipe tested in a hundred kitchens, and rated the same way in each, tells you more.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_compare, choice_linguist, result_cooccur, result_score):
    L.wrap_up(result_cooccur, result_score, choice_linguist, choice_compare)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
