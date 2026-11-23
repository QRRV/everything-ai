# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-word-relations. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-word-relations", "en")
    C = L.checks   # WordNet's senses and relations, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Three questions, about [Why language is hard for computers](/course/open/nlp-why-language-is-hard?lang=en) and [Repeating until something happens: while](/course/open/py-while?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_readings():
    quick_readings = L.choice(
        "quick-readings",
        question="In *Why language is hard for computers*, `readings` multiplied the meanings of the words of a sentence. A sentence has four words with more than one meaning: three with two each, and one with three. How many readings does it get?",
        options={
            "sum": "9: the meanings added up",
            "right": "24",
            "twelve": "12",
            "four": "4",
        },
        correct="right",
        why={
            "right": "2 × 2 × 2 × 3 = 24: each word multiplies the readings by its number of meanings. This lesson asks what those meanings are, and how a dictionary lists them.",
            "sum": "Each word multiplies the readings: every meaning of one word goes with every meaning of the others. See [Why language is hard for computers](/course/open/nlp-why-language-is-hard?lang=en).",
            "twelve": "That leaves out one of the three words with two meanings: 2 × 2 × 3 = 12. All four count: 2 × 2 × 2 × 3.",
            "four": "That counts the words with more than one meaning. The readings come from multiplying their meanings. See [Why language is hard for computers](/course/open/nlp-why-language-is-hard?lang=en).",
        },
    )
    quick_readings
    return (quick_readings,)


@app.cell(hide_code=True)
def quick_check_variation():
    quick_variation = L.choice(
        "quick-variation",
        question="*Why language is hard for computers* gave three reasons. Which was variation?",
        options={
            "ambiguity": "One form with several meanings",
            "right": "One meaning in many forms",
            "context": "A meaning that depends on who says it, and where",
        },
        correct="right",
        why={
            "right": "*colour* and *color* are one meaning in two spellings. Two different words can share one meaning too, such as *couch* and *sofa*: this lesson calls that synonymy.",
            "ambiguity": "That was ambiguity, the other way round: one word, such as *bank*, with several meanings. Variation was one meaning written in many forms, such as *colour* and *color*.",
            "context": "That was context. Variation was one meaning written in many forms, such as *colour* and *color*. See [Why language is hard for computers](/course/open/nlp-why-language-is-hard?lang=en).",
        },
    )
    quick_variation
    return (quick_variation,)


@app.cell(hide_code=True)
def quick_check_while():
    quick_while = L.choice(
        "quick-while",
        question="What is `n` after this code?\n\n```python\nn = 3\nwhile n < 20:\n    n = n * 2\n```",
        options={
            "twelve": "12, the last number below 20",
            "twenty": "20",
            "right": "24",
            "more": "48",
        },
        correct="right",
        why={
            "right": "3 becomes 6, 12 and then 24. Before every round, the loop asks the condition again: at 24 it is `False`, so the loop stops. Exercise 3 climbs with a `while` loop until there is nothing above.",
            "twelve": "At 12 the condition `n < 20` is still `True`, so the loop runs once more: 12 becomes 24. The loop stops only when the condition is `False`.",
            "twenty": "`n` doubles every round: 3, 6, 12, 24. It never equals 20; the loop stops at the first value that is not below 20. See [Repeating until something happens: while](/course/open/py-while?lang=en).",
            "more": "At 24 the condition `n < 20` is `False`, so the loop stops there and does not double once more. See [Repeating until something happens: while](/course/open/py-while?lang=en).",
        },
    )
    quick_while
    return (quick_while,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Search for *automobile*, and a good search engine also finds pages that only say *car*: the two words share a meaning. Ask whether a poodle is an animal, and a program has to know that a poodle is a kind of dog, and a dog a kind of animal. Word embeddings, as in *Word embeddings: neighbors and analogies*, put related words close together, but they do not say how the words relate. Daniel Jurafsky and James H. Martin give an example in *Speech and Language Processing*: "if a user asks a dialogue agent to turn up the music, it would be unfortunate to instead turn it down." And they warn that "in embedding models like word2vec, antonyms are easily confused with each other, because often one of the closest words in embedding space to a word (e.g., up) is its antonym (e.g., down)."

    This lesson looks at what dictionaries write down about meaning: the meanings of a word, the words that share a meaning, what something is a kind of, and what it is a part of.

    **An everyday example.** A dictionary numbers the meanings of a word, 1, 2, 3, each with its own definition, and a thesaurus gives the words with the same meaning, one list per meaning.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_one = L.guess(
        "guess",
        question="WordNet is a large database of the meanings of English words. It lists 127,846 different words, from single words such as *dog* to names and terms of several words. What share of them has exactly one meaning in WordNet?",
        options={
            "fifth": "About 20%",
            "half": "About half",
            "right": "About 80%",
            "nearly": "About 95%",
        },
        reveal="**About 80%**: 102,341 of the 127,846 words, 80.05%, have one meaning. Most of them are rare words, names and terms of several words. The words people use most have many: *break* has 75, *cut* 70 and *run* 57. In exercise 2 you count them yourself.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Senses

    Jurafsky and Martin: "A sense (or word sense) is a discrete representation of one aspect of the meaning of a word." A dictionary lists the senses of a word one by one, each with a definition. This lesson uses the senses of WordNet, a database of English that the psychologist George A. Miller and colleagues began at Princeton University in the mid-1980s, in the open version that the Open English WordNet team keeps up to date, the edition of 2025. A later lesson, *WordNet: a network of meanings*, opens it up.

    WordNet names every sense after a word, its part of speech and a number: `bank.n.01` is the first noun sense of *bank*, with *n* for noun (*v* is a verb, *a* and *s* an adjective, *r* an adverb). A sense keeps the name of the first word WordNet lists for it, so the bank for money is called `depository_financial_institution.n.01`. *bank* has 18 senses: 10 as a noun and 8 as a verb. The first five nouns:

    | Sense | Definition in WordNet |
    |---|---|
    | `bank.n.01` | *sloping land (especially the slope beside a body of water)* |
    | `depository_financial_institution.n.01` | *a financial institution that accepts deposits and channels the money into lending activities* |
    | `bank.n.03` | *a long ridge or pile* |
    | `bank.n.04` | *an arrangement of similar objects in a row or in tiers* |
    | `bank.n.05` | *a supply or stock held in reserve for future use (especially in emergencies)* |

    ### Polysemy

    A word with more than one sense is **polysemous**, from the Greek *poly*, many, and *sema*, sign; the property is called **polysemy**. Jurafsky and Martin use the word for every word with more than one sense. Others use it only for senses that are related, such as the bank as a company and the bank as its building, and call senses that have nothing to do with each other **homonymy**, such as the bank of a river and the bank for money. Related senses often come in patterns: the organization and its building go together for *school*, *university* and *hospital* too, Jurafsky and Martin note.

    ### Synonymy

    Two words with the same sense, or nearly the same, are **synonyms**: *couch* and *sofa*. Jurafsky and Martin stress that "synonymy is actually a relationship between senses rather than words." *big* and *large* can swap in *How big is that plane?*, but not in *a big sister*: *big* has a sense, older, that *large* lacks. WordNet gives every sense all of its words, so synonyms are words that share a name: `sofa.n.01` has the words *sofa*, *couch* and *lounge*, and the Dutch words *bank*, *canapé* and *sofa*.

    ### Hyponymy

    In their words: "A word (or sense) is a hyponym of another word or sense if the first is more specific, denoting a subclass of the other." Their example: "dog is a hyponym of animal". The more general sense is the **hypernym**: *animal* is a hypernym of *dog*. The relation is **hyponymy**, also called IS-A: a dog is an animal. The relation usually carries on up the ladder; Jurafsky and Martin call it transitive: "if A is a hyponym of B and B is a hyponym of C, then A is a hyponym of C." In WordNet, almost every noun and verb sense has one or more hypernyms. Taking the first one, step by step, `dog.n.01` climbs in 13 steps to `entity.n.01`, the one noun sense with no hypernym: canine, carnivore, placental, mammal, vertebrate, chordate, animal, organism, living thing, whole, object, physical entity, entity.

    ### Meronymy

    Jurafsky and Martin again: "Another common relation is meronymy, the part-whole relation. A leg is part of a chair; a wheel is part of a car." The part is a **meronym** of the whole, and the whole a **holonym** of the part. WordNet keeps three kinds apart: a part, such as the accelerator, one of the 30 parts of `car.n.01`; a member, such as a sheep of a flock; and a substance, as water is made of hydrogen and oxygen.

    Last, **antonyms** have opposite senses: *hot* and *cold*, *up* and *down*. WordNet ties an antonym to a word in a sense: *hot* in `hot.a.01` has *cold* in `cold.a.01`.

    The picture shows one sense of WordNet with its neighbors.
    """)
    return


@app.cell(hide_code=True)
def idea_dog():
    mo.Html(C.dog_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **word sense**, sense | one meaning of a word, as a dictionary numbers them; WordNet names one such as `bank.n.01` |
        | **polysemy**, polysemous | a word having more than one sense |
        | **homonymy** | for some authors: senses of one word that have nothing to do with each other |
        | **synonymy**, synonym | two words with the same sense, or nearly the same |
        | **hyponymy**, hyponym, hypernym | the IS-A relation: a hyponym is a kind of its hypernym, the more general sense |
        | **meronymy**, meronym, holonym | the part-whole relation: a meronym is a part, member or substance of its holonym |
        | **antonym** | a word with the opposite sense |
        | `wordnet.senses(word)` | the names of the senses of an English word: nouns first, then verbs, adjectives and adverbs, each from the most common sense on |
        | `wordnet.words(name)` | the words of a sense, such as `["sofa", "couch", "lounge"]` |
        | `wordnet.synset(name)` | a sense with what WordNet holds about it: `.definition`, `.hypernyms`, `.parts`, `.members`, `.substances` and `.dutch`; WordNet calls a sense with its words a synset |
        """),
        L.watch_out("Dictionaries do not agree on how many senses a word has. Jurafsky and Martin: \"Dictionaries tend to use many fine-grained senses so as to capture subtle meaning differences.\" And: \"For computational purposes, we often don't need these fine distinctions, so we often group or cluster the senses.\" WordNet makes fine distinctions too."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. To find the senses of a word, look it up: `wordnet.senses(word)` gives their names.
    2. Two words are synonyms where they share a sense: the same name among the senses of both.
    3. To climb from a sense to the most general one, take its first hypernym, then the first hypernym of that, and so on, until there is none.
    4. For a whole and its parts, look at `.parts`, `.members` and `.substances` of the sense.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The code looks up the senses of *couch*, with the words of each sense. Then it takes the first sense and prints its definition, its hypernyms and its Dutch words.
    """)
    return


@app.cell
def example_cell():
    from everything_ai import wordnet

    for name in wordnet.senses("couch"):
        print(name, wordnet.words(name))
    sofa = wordnet.synset("sofa.n.01")
    print(sofa.definition)
    print(sofa.hypernyms, sofa.dutch)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *couch* has four senses: three nouns and one verb. The first, `sofa.n.01`, has the words *sofa*, *couch* and *lounge*: three synonyms. The other two nouns, `couch.n.02` and `couch.n.03`, have only *couch*. The verb `frame.v.04`, *formulate in a particular style or language*, has *frame*, *redact*, *cast*, *put* and *couch*. The definition of `sofa.n.01` is *an upholstered seat for more than one person*; its hypernym is `seat.n.03`, and its Dutch words are *bank*, *canapé* and *sofa*. So the Dutch *bank* is polysemous too: a seat, and a place for money.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which relation?

    Each line holds two words. Put it under the relation between them.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_relations = L.sort("relations")
    sort_relations
    return (sort_relations,)


@app.cell(hide_code=True)
def check_sort(sort_relations):
    result_sort = L.check("relations", sort_relations)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_after_sort():
    mo.md(r"""
    In WordNet, *doctor* and *physician* are two words of `doctor.n.01`, and *bicycle* and *bike* of `bicycle.n.01`. `sparrow.n.01` climbs to `bird.n.01` in two steps, and `salmon.n.01` to `fish.n.01` in five. `finger.n.01` is one of the parts of `hand.n.01`, and `tree.n.01` is a member of `forest.n.01`. *mouse* has six senses, among them `mouse.n.01`, the animal, and `mouse.n.04`, the device; *bat* has ten, among them `bat.n.01`, the animal, and `cricket_bat.n.01`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_count():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. count_senses: how many senses per word?

        Write `count_senses(pairs)`. `pairs` is a list of pairs `(word, name)`: a word and the name of one of its senses, as WordNet gives them, one pair for every word of every sense. Give back a dict from each word to its number of senses: the number of pairs with that word.

        For `count_senses([("bank", "bank.n.01"), ("couch", "sofa.n.01"), ("bank", "depository_financial_institution.n.01"), ("sofa", "sofa.n.01")])` the result is `{"bank": 2, "couch": 1, "sofa": 1}`. The box starts an empty dict `counts` and already goes over the pairs as `word, name`; replace the `...`. The check calls that and `count_senses([])`, which should give `{}`, then cases of its own: one word with three senses, two words that share a sense, words that come back after other words, and `C.pairs_for(["break", "bank", "poodle"])`, all senses of three words of WordNet. The picture gives your function all 184,845 pairs of WordNet.
        """),
        L.task("Give back a dict from each word in pairs to its number of pairs."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count_senses", lines=9)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count_senses", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count_senses",
        [
            "Inside the loop, an `if` with an `else:` under it: one way for a word that came by before, one for a word that is new. Both ways set `counts[word]`: one more than before, or a first count.",
            "The first time *bank* comes by, `counts` has no key `\"bank\"` yet, and reading `counts[\"bank\"]` then stops with a KeyError. What can you ask about `counts` before you read the number of a word?",
            r"`        if ___:`",
        ],
        why="`if word in counts:` with `counts[word] = counts[word] + 1` under it, and `else:` with `counts[word] = 1` under that. For the example: *bank* gets 1, then *couch* 1, then *bank* 2, then *sofa* 1, so the result is `{\"bank\": 2, \"couch\": 1, \"sofa\": 1}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_count():
    mo.md(r"""
    In the picture, your function counted the senses of all 127,846 words of WordNet. 80% have one sense, as in the guess, and the words with the most are short, common words: *break*, *cut*, *run*, *play*, *make*, *light*. GloVe, a set of word embeddings learned from a large collection of texts, lists its 50,000 words from the most to the least frequent in those texts, and that order shows the pattern: the 736 words of WordNet among GloVe's 1,000 most frequent words have 8.25 senses on average, the words of WordNet among the next 4,000 have 4.94, among the next 15,000 2.84 and among the next 30,000 1.94. Words that are not among GloVe's 50,000 have 1.17. GloVe writes every word in lowercase, so a word with a capital, such as *Monday*, counts among the words GloVe does not have. In 1945 the linguist George Kingsley Zipf described this link: the more often a word is used, the more meanings it tends to have.

    Averaged per part of speech, a noun of WordNet has 1.26 senses and a verb 2.17. For WordNet 3.0, the version of 2006, Jurafsky and Martin give the same picture: "The average noun has 1.23 senses, and the average verb has 2.16 senses."
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chain():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. chain: up the ladder

        Write `chain(name, parent)`. `parent` is a dict from a sense to its first hypernym; the top has none, so it is not a key of `parent`. Give back the list from `name` up to the top: `name`, its parent, the parent of that, and so on, up to a sense that has no parent.

        For `chain("poodle", {"poodle": "dog", "dog": "animal"})` the result is `["poodle", "dog", "animal"]`. The box starts the list `out` with `name`; replace the `...` with a `while` loop. The check calls that and `chain("animal", {"poodle": "dog", "dog": "animal"})`, which should give `["animal"]`, then cases of its own: a longer ladder, a name with no parent at all, a dict with branches, and three senses of WordNet with `C.PARENT`, the first hypernym of each of the 85,586 senses of WordNet that have one.
        """),
        L.task("Give back the list from name up to the top, each next item the parent of the one before."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chain():
    box_chain = L.code_box("chain", lines=6)
    box_chain
    return (box_chain,)


@app.cell(hide_code=True)
def check_chain(box_chain):
    result_chain = L.check("chain", box_chain.value)
    result_chain
    return (result_chain,)


@app.cell(hide_code=True)
def hints_chain():
    L.hints(
        "chain",
        [
            "Use `while`, with a condition that asks whether there is still a step up, and in the loop append one parent to `out`, with `parent[...]`. After the loop, `return out`.",
            "If the loop does not change `name`, a condition about `name` never changes either. Which item of `out` is the one whose parent comes next, and how do you write it?",
            r"`    while ___ in parent:`",
        ],
        why="`while out[-1] in parent:` with `out.append(parent[out[-1]])` under it: `out[-1]` is the last name so far, and the loop stops at the first name that has no parent. For the example: `out` is `[\"poodle\"]`, then `[\"poodle\", \"dog\"]`, then `[\"poodle\", \"dog\", \"animal\"]`, and *animal* is not a key, so the loop stops.",
    )
    return


@app.cell(hide_code=True)
def text_after_chain():
    mo.md(r"""
    In the picture, your function climbed from four senses. The ladders of `poodle.n.01`, 15 rungs, and `cat.n.01`, 14 rungs, meet at `carnivore.n.01`: from there up they are the same. The two senses of *bank* share nothing below `entity.n.01`: the bank of a river is a slope, a geological formation and an object; the bank for money is an institution, an organization, a social group and an abstraction. Jurafsky and Martin show the same for two senses of *bass*, the singer and the instrument: "the chains are completely distinct, only converging at the very abstract level whole, unit." How far up two ladders meet says something about how alike two senses are: *WordNet: a network of meanings* makes a measure of it.
    """)
    return


@app.cell(hide_code=True)
def text_explore():
    mo.md(r"""
    ### Try it: the senses of a word

    Type a word, in English or in Dutch. The table gives its senses in WordNet, with the words of each sense, the Dutch words and the definition. Under it: the ladder of the sense whose number you choose, with its parts, members, substances and antonyms, and the more specific senses below it. Try *head*, *spring* or *bass*, or choose Nederlands and try *bank* or *blad*. Some links to Dutch words are wrong: `head.n.01`, the head of a body, has the Dutch words *baas* and *leider*, boss and leader, which belong to another sense of *head*.
    """)
    return


@app.cell(hide_code=True)
def explore_controls():
    explore_word = mo.ui.text(value="bank", label="Word")
    explore_language = mo.ui.radio(options=["English", "Nederlands"], value="English", label="Language", inline=True)
    explore_number = mo.ui.number(start=1, stop=99, step=1, value=1, label="Sense for the ladder")
    mo.hstack([explore_word, explore_language, explore_number], justify="start", wrap=True)
    return explore_language, explore_number, explore_word


@app.cell(hide_code=True)
def explore_look(explore_language, explore_number, explore_word):
    mo.Html(C.explore_picture(explore_word.value, explore_language.value, explore_number.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Why language is hard for computers](/course/open/nlp-why-language-is-hard?lang=en) counted the readings of words with more than one meaning. Here the meanings have names: the senses of WordNet, 18 for *bank*. Most words have one sense, and the words used most have many. Several words can share one sense: synonymy. A sense is a kind of a more general one, up a ladder to `entity.n.01`: hyponymy, which your `while` loop climbed. And a part, member or substance of a whole: meronymy. The next lesson of the domain, *WordNet: a network of meanings*, joins these relations into one network, and measures how far apart two senses are.

    **An everyday example.** The thesaurus in a word processor gives the synonyms of *big* in groups: *large* and *huge* for size, *older* and *grown-up* for a big sister.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_chain, result_count, result_sort):
    L.wrap_up(result_sort, result_count, result_chain)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
