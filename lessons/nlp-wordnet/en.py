# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-wordnet. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-wordnet", "en")
    C = L.checks   # WordNet as a graph, the kitchen lexicon, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions, about [Senses and relations between words](/course/open/nlp-word-relations?lang=en) and [Graphs: nodes, edges and paths](/course/open/math-graphs?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_meet():
    quick_meet = L.choice(
        "quick-meet",
        question="In *Senses and relations between words*, your `chain` climbed the ladders of first hypernyms of `poodle.n.01` and `cat.n.01`. At which sense did the two ladders meet?",
        options={
            "entity": "`entity.n.01`, only at the very top",
            "animal": "`animal.n.01`",
            "right": "`carnivore.n.01`",
            "dog": "`dog.n.01`",
        },
        correct="right",
        why={
            "right": "From `carnivore.n.01` up, the two ladders are the same. How low two senses meet says how alike they are: this lesson turns it into a number.",
            "entity": "That was the river bank and the bank for money, which share nothing below the top. A poodle and a cat are both carnivores. See [Senses and relations between words](/course/open/nlp-word-relations?lang=en).",
            "animal": "Both ladders pass `animal.n.01`, but they meet lower: a poodle and a cat are both carnivores, and from `carnivore.n.01` up the ladders are the same.",
            "dog": "A cat is not a kind of dog, so `dog.n.01` is only on the ladder of `poodle.n.01`. The two meet higher up, at `carnivore.n.01`.",
        },
    )
    quick_meet
    return (quick_meet,)


@app.cell(hide_code=True)
def quick_check_bfs():
    quick_bfs = L.choice(
        "quick-bfs",
        question="A directed graph as an adjacency list, with an edge from each node to each node in its list:\n\n```python\ngraph = {\"A\": [\"B\", \"E\"], \"B\": [\"C\"], \"C\": [\"D\"], \"E\": [\"D\"], \"D\": []}\n```\n\nBreadth-first search starts at A. Which path to D does it find?",
        options={
            "first": "A, B, C, D, because B comes first in the list of A",
            "right": "A, E, D",
            "direct": "A, D",
            "none": "None: the list of D is empty",
        },
        correct="right",
        why={
            "right": "Breadth-first search tries every path of 1 edge, then every path of 2 edges, so the first path that reaches D has the fewest edges: A, E, D. In this lesson, the same search climbs from a sense to all of its hypernyms.",
            "first": "Breadth-first search does not follow B to the end first: it tries every path of 1 edge, then of 2, so it reaches D through E after 2 edges, before the path through B and C. See [Graphs: nodes, edges and paths](/course/open/math-graphs?lang=en).",
            "direct": "A has no edge to D: the list of A holds B and E. The shortest path takes 2 edges.",
            "none": "The empty list says that no edge leaves D. Edges do arrive at D, from C and from E, so D can be reached.",
        },
    )
    quick_bfs
    return (quick_bfs,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Is a cat like a dog? Is coffee like a cup? Daniel Jurafsky and James H. Martin tell the two questions apart in *Speech and Language Processing*: "Cat is not a synonym of dog, but cats and dogs are certainly similar words." And: "Coffee is not similar to cup; they share practically no features (coffee is a plant or a beverage, while a cup is a manufactured object with a particular shape). But coffee and cup are clearly related; they are associated by co-participating in an everyday event (the event of drinking coffee out of a cup)." Word embeddings, as in *Word embeddings: neighbors and analogies*, mix the two. GloVe, a set of word embeddings learned from a large collection of texts, gives every word a vector, and the cosine similarity of two vectors, at most 1, says how much the two point the same way: *car* gets 0.92 with *truck*, and 0.85 with *driver*, though a driver is a person and a truck a vehicle. A shop that offers tea when the coffee is sold out needs words that are alike, and a cup would not help.

    WordNet holds what things are a kind of. This lesson joins its senses into one network and measures how far apart two senses are in it.

    **An everyday example.** A supermarket puts tea next to coffee, on the shelf of drinks, and cups in another aisle, though you drink coffee from a cup. The shelves follow what a thing is; a shopping list follows what goes together.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_one = L.guess(
        "guess",
        question="In WordNet, each noun sense has a more general one, up to one top. Take two noun senses at random, and walk from the one up to the nearest sense above both, and down to the other. How many steps does that walk take, on average?",
        options={
            "four": "About 4",
            "eight": "About 8",
            "right": "About 14",
            "many": "More than 30",
        },
        reveal="**About 14**: over 2,000 random pairs of noun senses, the shortest walk takes 13.66 steps on average, from 1 to 24. Most pairs meet only high up, near the top. Senses that are alike meet much lower: `coffee.n.01` and `tea.n.01` are 2 steps apart. Exercise 3 turns the number of steps into a similarity.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### WordNet, a lexicon

    Jurafsky and Martin: "The most commonly used resource for sense relations in English and many other languages is the WordNet lexical database (Fellbaum, 1998)." WordNet is a **lexicon**: a list of words with information about each one, a dictionary made for computers. For every word it gives its senses, and for every sense its words, a definition and its relations to other senses. In their words again: "The set of near-synonyms for a WordNet sense is called a synset (for synonym set); synsets are an important primitive in WordNet." `sofa.n.01` of *Senses and relations between words* is a **synset** with three words, *sofa*, *couch* and *lounge*. The relations of WordNet, such as hyponymy and meronymy, join synsets. The edition of Open English WordNet that the course uses has 107,425 synsets.

    ### A graph of meanings

    Take each synset as a node, and each link to a hypernym as an edge, with a direction: from the more specific sense to the more general one. Then WordNet is a directed graph, as in [Graphs: nodes, edges and paths](/course/open/math-graphs?lang=en), with 87,970 such edges. The course stores it as an adjacency list: `C.HYPERNYMS` is a dict from each of the 107,425 synsets to the list of its hypernyms. Every noun sense climbs to `entity.n.01`, the one noun sense without a hypernym. The verbs climb to one of 8 tops: 12,898 of the 13,811 verb senses climb to `act.v.01`, others to `exist.v.01` or `have.v.01`, for example. The 21,830 adjective and adverb senses have no hypernyms at all: their lists are empty.

    A sense can have more than one hypernym: 2,298 noun senses and 38 verb senses do. `dog.n.01` has two, `canine.n.02` and `domestic_animal.n.01`. So above a sense, its hypernyms, theirs and so on form a small graph, and one ladder misses part of it. Along first hypernyms, as `chain` climbed in *Senses and relations between words*, `animal.n.01` lies 7 steps above `dog.n.01`; through `domestic_animal.n.01` it is 2. Breadth-first search finds every sense above a sense, each with its fewest steps.

    ### Path similarity

    Two senses that are alike meet low. To measure it, take the shortest path between them through a hypernym they share: up from one sense to that hypernym, and down to the other. One of the two can itself be that hypernym: a sense and its own hypernym are 1 step apart. For `coffee.n.01` and `tea.n.01`, the path goes through `beverage.n.01` and has 2 steps; for `coffee.n.01` and `cup.n.01`, it goes through `physical_entity.n.01` and has 11, 5 up from coffee and 6 up from cup. NLTK, a Python library for language, turns the number of steps into a **path similarity**:

    $$\text{path similarity} = \frac{1}{\text{steps} + 1}$$

    A sense and itself are 0 steps apart and get 1; NLTK's documentation: "A score of 1 represents identity i.e. comparing a sense with itself will return 1." Coffee and tea get 1/3, coffee and cup 1/12. Two senses that share no hypernym, such as two verbs below different tops, have no path between them: then the path similarity in this lesson is `None`. NLTK itself adds a made-up top above all verb tops by default, and then gives such verbs a small similarity; with that made-up top switched off, it gives `None` too. The idea is old. Jurafsky and Martin describe how M. Ross Quillian, in 1968 and 1969, "proposed a graph-based approach to language processing", with the meanings of words in a network, and chose between the senses of a word "by finding the shortest path between senses in the graph."
    """)
    return


@app.cell(hide_code=True)
def idea_paths():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_lexicon():
    mo.md(r"""
    ### A lexicon of your own

    A small lexicon of your own is built the same way: write down the hypernyms of every word, in a dict. The kitchen lexicon of this lesson has 12 words: one top, *thing*, and 11 words below it, each with one hypernym.
    """)
    return


@app.cell(hide_code=True)
def idea_kitchen():
    mo.Html(C.kitchen_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **WordNet** | a large lexicon of English: senses with their words, definitions and relations |
        | **lexicon** | a list of words with information about each one, such as its senses or its hypernyms; a dictionary made for computers |
        | **synset** | a set of words that share one sense in WordNet, named such as `sofa.n.01`; the relations of WordNet join synsets |
        | **path similarity** | 1 / (steps + 1), for the shortest path between two senses through a hypernym they share; `None` without one |
        | **cosine similarity** | how much two vectors point the same way, at most 1; word embeddings give words that often have the same words around them a high one |
        | `C.HYPERNYMS` | the adjacency list of WordNet: a dict from every synset to the list of its hypernyms, `[]` for a sense without one |
        | `wordnet.synset(name).hypernyms` | the hypernyms of one synset, as in *Senses and relations between words* |
        | `min(numbers)` | the smallest number in a list: `min([5, 3, 4])` gives `3`; with an empty list, Python stops with a ValueError |
        """),
        L.watch_out("Every step counts the same, but WordNet splits some parts finer than others. `cat.n.01` lies 7 steps below `animal.n.01`: feline, carnivore, placental, mammal, vertebrate, chordate, animal. `coffee.n.01` lies 2 steps below `food.n.01`. Biologists sort animals into many levels, and WordNet follows them; path similarity counts each step the same, wherever it is."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Store a lexicon as an adjacency list: a dict from each sense to the list of its hypernyms, with `[]` for a top.
    2. To find every sense above a sense with its fewest steps, use breadth-first search upward: the sense itself is 0 steps up, and a hypernym of a sense that is *k* steps up is *k* + 1 steps up, unless the search reached it before.
    3. For two senses, take the senses that both reach, each sense counting itself at 0 steps up. For each, add its steps up from the one sense and from the other: the smallest sum is the length of the shortest path.
    4. The path similarity is 1 / (steps + 1). Without a shared hypernym, the result is `None`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The code stores the kitchen lexicon as an adjacency list. `ladder` climbs from a word along its first hypernym, as `chain` did, and keeps the steps up of every word it passes in a dict. Every word below the top has one hypernym, so one ladder finds every hypernym of a word. Last, the code prints the hypernyms of `beverage.n.01` in WordNet, a sense of *drink* as a noun: *any liquid suitable for drinking*.
    """)
    return


@app.cell
def example_cell():
    from everything_ai import wordnet

    kitchen = {
        "thing": [],
        "utensil": ["thing"],
        "food": ["thing"],
        "cutlery": ["utensil"],
        "pan": ["utensil"],
        "drink": ["food"],
        "soup": ["food"],
        "fork": ["cutlery"],
        "spoon": ["cutlery"],
        "knife": ["cutlery"],
        "tea": ["drink"],
        "coffee": ["drink"],
    }

    def ladder(word, hypernyms):
        steps = {word: 0}
        while len(hypernyms[word]) > 0:
            up = hypernyms[word][0]
            steps[up] = steps[word] + 1
            word = up
        return steps

    print(ladder("coffee", kitchen))
    print(ladder("soup", kitchen))
    print(wordnet.synset("beverage.n.01").hypernyms)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *coffee* climbs to *drink*, *food* and *thing*; *soup* to *food* and *thing*. The two share two hypernyms: *food*, 2 steps up from *coffee* and 1 from *soup*, 3 together, and *thing*, 3 and 2, 5 together. The shortest path, coffee, drink, food, soup, has 3 steps, so the path similarity is 1/(3 + 1) = 0.25. WordNet gives a drink two hypernyms: `beverage.n.01` is both a `food.n.01` and a `liquid.n.01`. A ladder of first hypernyms misses the second; exercise 2 finds them all.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which path similarity?

    Each line holds two words of the kitchen lexicon. Count the steps of the shortest path between them, and put the pair under its path similarity.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_similarity = L.sort("similarity")
    sort_similarity
    return (sort_similarity,)


@app.cell(hide_code=True)
def check_sort(sort_similarity):
    result_sort = L.check("similarity", sort_similarity)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_after_sort():
    mo.md(r"""
    *cutlery* is the hypernym of *fork* itself: 1 step, 1/2. *spoon* and *knife* hang from *cutlery*, 1 step up and 1 down: 1/3. *knife* and *pan* meet at *utensil*, 2 steps up from *knife* and 1 from *pan*: 1/4. *pan* and *soup* meet only at the top, *thing*, 2 steps up from each: 1/5. The lower two words meet, the higher their path similarity.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ancestors():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. ancestors: every sense above a sense

        Write `ancestors(name, hypernyms)`. `hypernyms` is an adjacency list: a dict from every sense to the list of its hypernyms, `[]` for a top. Give back a dict from `name` and every sense above it to its fewest steps up from `name`.

        For `ancestors("poodle", {"poodle": ["dog"], "dog": ["animal"], "cat": ["animal"], "animal": []})` the result is `{"poodle": 0, "dog": 1, "animal": 2}`. The box starts the dict `steps` with `name` at 0 steps and the `deque` `queue` with `name`, and takes the next sense from the front of the queue in every round, as breadth-first search does; replace the `...`. The check calls that and `ancestors("animal", ...)` with the same dict, which should give `{"animal": 0}`, then cases of its own: a sense with two hypernyms, a sense that two paths of different lengths reach, and three senses of WordNet with `C.HYPERNYMS`.
        """),
        L.task("Give back a dict from name and every sense above it to its fewest steps up."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ancestors():
    box_ancestors = L.code_box("ancestors", lines=13)
    box_ancestors
    return (box_ancestors,)


@app.cell(hide_code=True)
def check_ancestors(box_ancestors):
    result_ancestors = L.check("ancestors", box_ancestors.value)
    result_ancestors
    return (result_ancestors,)


@app.cell(hide_code=True)
def hints_ancestors():
    L.hints(
        "ancestors",
        [
            "In the loop, go over the hypernyms of `node`, with `for up in hypernyms[node]:`. A hypernym of `node` is one step further up than `node`: give it its steps in `steps`, and put it in the queue, so that the search goes on from there.",
            "Breadth-first search reaches a sense first along a path with the fewest steps. When it reaches the same sense again, along a longer path, its number should stay as it is. What can you ask about `steps` before you give `up` its number?",
            r"`            if ___:`",
        ],
        why="`for up in hypernyms[node]:` with `if up not in steps:` under it, and under that `steps[up] = steps[node] + 1` and `queue.append(up)`. For the example: *poodle* is 0; *dog* gets 1 and joins the queue; then *animal* gets 2. *animal* has no hypernyms, so the queue empties and the loop stops.",
    )
    return


@app.cell(hide_code=True)
def text_after_ancestors():
    mo.md(r"""
    In the picture, your function climbed from `dog.n.01` and from `domestic_cat.n.01`, the house cat. Each has two hypernyms, and both have `domestic_animal.n.01` among them: there the dog and the house cat meet, 1 step up from each, 2 steps together. Along first hypernyms they would meet only at `carnivore.n.01`, as the poodle and the cat did. `dog.n.01` has 15 senses above it, from `canine.n.02` and `domestic_animal.n.01` at 1 step to `entity.n.01` at 6.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_similarity():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. path_similarity: how alike are two senses?

        Write `path_similarity(a, b, hypernyms)`: the path similarity of the senses `a` and `b`, 1 / (steps + 1) for the shortest path between them through a hypernym they share, or `None` when they share none. The box may use `ancestors`: the course's version, which works as yours should. It starts `up_a` and `up_b`, the dicts of `ancestors` for both senses, and an empty list `steps`. Replace the `...` with a loop that puts in `steps` the number of steps of the path through each shared hypernym, and then the two `return` lines: one for no shared hypernym, one for the path similarity.

        For `path_similarity("poodle", "cat", {"poodle": ["dog"], "dog": ["animal"], "cat": ["animal"], "animal": []})` the result is `0.25`: poodle, dog, animal, cat is 3 steps. The check calls that and `path_similarity("cat", "cat", ...)` with the same dict, which should give `1.0`, then cases of its own: two pairs that share more than one hypernym, two senses with no hypernym in common, and four pairs of WordNet with `C.HYPERNYMS`.
        """),
        L.task("Give back 1 / (steps + 1) for the shortest path through a shared hypernym, or None when a and b share none."),
    ])
    return


@app.cell(hide_code=True)
def exercise_similarity():
    box_similarity = L.code_box("path_similarity", lines=11)
    box_similarity
    return (box_similarity,)


@app.cell(hide_code=True)
def check_similarity(box_similarity):
    result_similarity = L.check("path_similarity", box_similarity.value)
    result_similarity
    return (result_similarity,)


@app.cell(hide_code=True)
def hints_similarity():
    L.hints(
        "path_similarity",
        [
            "Go over the senses of `up_a` with a `for` loop, and ask for each whether it is in `up_b` too: then the two share it. After the loop: when `steps` is empty, `return None`; otherwise give back the path similarity of the shortest path.",
            "A path through a shared hypernym goes up from `a` to that hypernym, and down from there to `b`. Which two dicts tell how many steps each part takes?",
            r"`            steps.append(___)`",
        ],
        why="`for name in up_a:` with `if name in up_b:` and `steps.append(up_a[name] + up_b[name])` under it; then `if len(steps) == 0:` with `return None`, and last `return 1 / (min(steps) + 1)`. For the example: *poodle* and *cat* share *animal*, 2 steps up from *poodle* and 1 from *cat*, so `steps` is `[3]` and the path similarity 1/4 = 0.25.",
    )
    return


@app.cell(hide_code=True)
def text_after_similarity():
    mo.md(r"""
    In the picture, your function measured eight pairs of senses. The four that are alike get a path similarity of 0.25 or 0.33; the four that are only related get 0.08 or less, because their paths meet near the top: a teacher and a school share nothing below `entity.n.01`. GloVe gives both kinds a high cosine similarity: *car* and *driver* get 0.85, almost as much as *car* and *truck*. Word embeddings learn from the words around a word, and a driver comes up in the same texts as a car. *coffee* and *cup* get only 0.36 in GloVe. GloVe has one vector for all senses of a word together, and the nearest words of *cup* in GloVe are *champions*, *championship* and *tournament*: the cups of sports.
    """)
    return


@app.cell(hide_code=True)
def text_explore_lexicon():
    mo.md(r"""
    ### Try it: a lexicon of your own

    Change the lexicon in the cell below: add words, such as *plate* under *utensil* or *bread* under *food*, give a word a second hypernym, such as *drink* also under a new word *liquid*, or start a lexicon about something else. Then choose two words: the picture draws your lexicon from its tops down, with the shortest path between them.
    """)
    return


@app.cell
def explore_lexicon():
    my_lexicon = {
        "thing": [],
        "utensil": ["thing"],
        "food": ["thing"],
        "cutlery": ["utensil"],
        "pan": ["utensil"],
        "drink": ["food"],
        "soup": ["food"],
        "fork": ["cutlery"],
        "spoon": ["cutlery"],
        "knife": ["cutlery"],
        "tea": ["drink"],
        "coffee": ["drink"],
    }
    return (my_lexicon,)


@app.cell(hide_code=True)
def explore_menus(my_lexicon):
    _words = C.lexicon_words(my_lexicon)
    explore_first = mo.ui.dropdown(options=_words, value=C.first_choice(_words, L.lang), label="First word")
    explore_second = mo.ui.dropdown(options=_words, value=C.second_choice(_words, L.lang), label="Second word")
    mo.hstack([explore_first, explore_second], justify="start", wrap=True)
    return explore_first, explore_second


@app.cell(hide_code=True)
def explore_tree(explore_first, explore_second, my_lexicon):
    mo.Html(C.lexicon_picture(my_lexicon, explore_first.value, explore_second.value, L.lang))
    return


@app.cell(hide_code=True)
def text_explore_words():
    mo.md(r"""
    ### Try it: two words in WordNet

    Type two English words. The picture tries every noun sense of the first with every noun sense of the second, and every verb sense with every verb sense, and shows the pairs of senses that are most alike, with the path of the best, and the cosine similarity of the two words in GloVe. The best pair can be a sense you did not mean: for *coffee* and *cup*, it is `cup.n.06`, *a punch served in a pitcher instead of a punch bowl*: a drink, as coffee is. Try *doctor* and *nurse*, *doctor* and *hospital*, or *run* and *walk*.
    """)
    return


@app.cell(hide_code=True)
def explore_controls():
    explore_a = mo.ui.text(value="coffee", label="First word")
    explore_b = mo.ui.text(value="cup", label="Second word")
    mo.hstack([explore_a, explore_b], justify="start", wrap=True)
    return explore_a, explore_b


@app.cell(hide_code=True)
def explore_look(explore_a, explore_b):
    mo.Html(C.words_picture(explore_a.value, explore_b.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Senses and relations between words](/course/open/nlp-word-relations?lang=en) named the senses of WordNet and climbed one ladder of first hypernyms. Here the synsets and their hypernyms form a directed graph, stored as an adjacency list as in [Graphs: nodes, edges and paths](/course/open/math-graphs?lang=en), and breadth-first search finds every sense above a sense with its fewest steps. The shortest path through a shared hypernym gives a path similarity: high for senses that are alike, such as coffee and tea, and low for senses that only go together, such as a car and its driver, which GloVe puts close. A lexicon of your own works the same way; a later lesson, *A list of good and bad words*, builds a lexicon of words with their sentiment. The next lesson of the domain, *A vector for a whole sentence*, goes back to vectors, now for sentences.

    **An everyday example.** A library sorts its books by subject, with subjects inside subjects. Two books on one shelf are alike; two books in different halls meet only at the top: books.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_ancestors, result_similarity, result_sort):
    L.wrap_up(result_sort, result_ancestors, result_similarity)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
