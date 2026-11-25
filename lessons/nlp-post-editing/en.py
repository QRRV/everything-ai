# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-post-editing. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-post-editing", "en")
    C = L.checks   # the sentences, the translations by Qwen and by people, the memory, TER, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [How far apart are two words? Edit distance](/course/open/nlp-edit-distance?lang=en), and one about [Scoring a translation: BLEU and chrF](/course/open/nlp-bleu?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_wer():
    quick_wer = L.choice(
        "quick-wer",
        question="In *How far apart are two words? Edit distance*, the word error rate counts the insertions, deletions and substitutions between what someone said and what a program heard, as a share of the words said. Someone says *turn on the light*, and the program hears *turn the lights*. What is the word error rate?",
        options={
            "place": "75%, as only *turn* is at the same place in both",
            "right": "50%",
            "heard": "67%, for 2 edits on 3 words heard",
            "missing": "25%, for the missing *on*",
        },
        correct="right",
        why={
            "right": "Insert *on* and substitute *light* for *lights*: 2 edits on 4 words said, 50%. Counted place by place, *the* would stand against *on*, and every word after it would count as wrong; the edit distance lines the words up first. This lesson lines up a translation with a reference the same way, and adds one more kind of edit: moving a block of words.",
            "place": "Counted place by place, *the* stands against *on*, and every word after it counts as wrong. What does the edit distance do first, before it counts? See [How far apart are two words? Edit distance](/course/open/nlp-edit-distance?lang=en).",
            "heard": "The word error rate divides by the words that were said, not by the words that were heard. How many words did the person say? See [How far apart are two words? Edit distance](/course/open/nlp-edit-distance?lang=en).",
            "missing": "*on* is missing, and one more word differs: *lights* where the person said *light*. How many edits is that? See [How far apart are two words? Edit distance](/course/open/nlp-edit-distance?lang=en).",
        },
    )
    quick_wer
    return (quick_wer,)


@app.cell(hide_code=True)
def quick_check_bleu():
    quick_bleu = L.choice(
        "quick-bleu",
        question="The translation *take the next bus i will* has every word of the reference *i will take the next bus*, and n-gram precisions, as in *Scoring a translation: BLEU and chrF*, of 6/6, 4/5, 2/4 and 1/3. Why are the precisions of the longer n-grams lower?",
        options={
            "clip": "Clipping lets each word count only once",
            "few": "Six words have only three groups of four, so the share drops",
            "right": "Two blocks stand in each other's place",
            "short": "The brevity penalty lowers them",
        },
        correct="right",
        why={
            "right": "*i will* stands at the end and *take the next bus* at the front, so the groups of words that cross from one block into the other, such as *bus i*, are not in the reference. A person fixes it with one move: *i will* to the front. BLEU does not count that move; this lesson counts it as one edit.",
            "clip": "Clipping only matters for an n-gram that the translation has more often than the reference, and every word here occurs once. Which groups of words in a row does the reference lack? See [Scoring a translation: BLEU and chrF](/course/open/nlp-bleu?lang=en).",
            "few": "A precision is a share: 1 of 3 groups of four is the same share as 2 of 6 would be. Which groups of words in a row does the reference lack? See [Scoring a translation: BLEU and chrF](/course/open/nlp-bleu?lang=en).",
            "short": "The brevity penalty multiplies BLEU as a whole, not each precision, and the translation has as many words as the reference. Which groups of words in a row does the reference lack? See [Scoring a translation: BLEU and chrF](/course/open/nlp-bleu?lang=en).",
        },
    )
    quick_bleu
    return (quick_bleu,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    BLEU says how close a translation is to translations by people. A translator who gets a machine translation asks something else: how much work is it to fix? In a 1997 survey of machine translation, Martin Kay looked back on the move away from machines that translate alone, "in favor of one in which the task is apportioned between people and machines." For systems that translate whole texts, Kay wrote, "post-editing of the machine's output is generally taken for granted." In 2010, Mirko Plitt and François Masselot measured what that brings at Autodesk, a software company, with "twelve participants translating from English to French, Italian, German, and Spanish." Throughput is how much a translator gets done in an hour: "MT allowed translators to improve their throughput on average by 74%; in other words, MT saved 43% of the translation time." MT is short for machine translation. The gain differed per translator, "from 20% to 131%", and one result surprised them: "To our surprise, translation jobs contained a higher number of mistakes than post-editing jobs."

    In this lesson you count the edits a translation needs, with TER, and compare two starting points for a translator: a translation by Qwen3-0.6B, the course's chat model, and a translation memory, which looks up how a sentence that looks like the new one was translated before. The sentences come from Tatoeba: the 3,707 Dutch sentences of *Scoring a translation: BLEU and chrF*, without the 29 for which Qwen's translation uses a gendered pronoun, which the course leaves out, so 3,678. The memory holds 120,728 other Dutch sentences of Tatoeba, each with an English translation by people; there too, the course left out sentences with a gendered pronoun in a translation.

    **An everyday example.** Fixing a draft that a colleague wrote: you keep what is right, change a word here, move a sentence there. When the draft is good, that is faster than writing it all yourself; when it is bad, you rewrite it anyway.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_exact = L.guess(
        "guess",
        question="The course had Qwen3-0.6B translate the 3,678 Dutch sentences of this lesson into English, sentence by sentence, with the prompt *Translate this Dutch sentence into English. Reply with only the English translation.* Each sentence has two to 14 translations by people. For how many of the 3,678 is Qwen's translation, once its words are in small letters without punctuation, exactly one of the translations by people?",
        options={
            "almost": "Almost none",
            "tenth": "About one in ten",
            "right": "About three in ten",
            "most": "More than two thirds",
        },
        reveal="1,142 of the 3,678, or 31.0%. The word-for-word translations of *Translating by rules* managed 726. A translation that differs is not always wrong, as *Ik moet naar bed.* shows: Qwen wrote *I must go to bed.*, and people wrote *I have to go to bed.* and *I need to go to sleep.*",
    )
    guess_exact
    return (guess_exact,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Post-editing** is correcting a machine translation instead of translating from scratch: the translator keeps what is right and changes the rest. It is part of **computer-assisted translation** (CAT): a person translates, with software that helps. An older kind of help is a **translation memory**: every sentence a translator has translated, stored with its translation. For a new sentence, the memory looks up the stored sentence that comes closest, a **fuzzy match**, and shows its translation, which the translator post-edits like a machine translation. Tools show how close a match is as a percentage, with 100% for the same sentence, and each tool works it out in its own way. This lesson uses the edit distance of the words:

    $$\text{fuzzy} = 1 - \frac{\text{edit distance}}{\text{length of the longer sentence}}$$

    How much does a translation still need? In 2006, Matthew Snover and four colleagues proposed to count edits: "TER is defined as the minimum number of edits needed to change a hypothesis so that it exactly matches one of the references, normalized by the average length of the references." The **hypothesis** is the translation being scored. The edits are those of the word error rate, plus one: "Possible edits include the insertion, deletion, and substitution of single words as well as shifts of word sequences." A **shift** moves a block of words to another place, as a single edit: "All edits, including shifts of any number of words, by any distance, have equal cost."

    TER does not try every possible set of shifts. It is **greedy**: it takes the best step it can see, one at a time, without looking further ahead. "A greedy search is used to find the set of shifts, by repeatedly selecting the shift that most reduces the number of insertions, deletions and substitutions, until no more beneficial shifts remain." And a shift has to put words where they belong: "The shifted words must match the reference words in the destination position exactly." The edit distance counts the rest. Like the word error rate, TER is written times 100, lower is better, and it can go above 100 when a translation has many words too many.

    Translations by people that were made without seeing the machine translation say the same thing in their own words, and TER counts those differences as edits too. So Snover and colleagues also had people who speak the target language well make, for each machine translation, "a new reference translation targeted for this system output by editing the hypothesis until it is both fluent and has the same meaning as the reference(s)." TER against such a **targeted reference** is **HTER**, human-targeted TER: the fewest edits from the translation to its post-edit, divided, as for TER, by the average length of the references that were made without seeing the machine translation. It costs time: "We found that annotators took an average of 3 to 7 minutes per sentence to provide a good targeted reference." Annotators are the people who did it.
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
    | **post-editing** | correcting a machine translation, and keeping what is right, instead of translating from scratch |
    | **computer-assisted translation** (CAT) | translation by a person, with software that helps, such as a translation memory |
    | **translation memory** | sentences translated before, each stored with its translation |
    | **fuzzy match** | a stored sentence that looks like a new sentence, with a score for how close it is; 100% is the same sentence |
    | **hypothesis** | the translation that TER scores |
    | **shift** | moving a block of words to another place in a sentence, as one edit |
    | **greedy** | taking the best step in sight, one at a time, without looking further ahead |
    | **TER** | translation edit rate: the fewest edits, shifts included, to the closest reference, divided by the average length of the references, times 100 |
    | **targeted reference** | a reference made by post-editing the translation itself |
    | **HTER** | TER against a targeted reference: the fewest edits from a translation to its post-edit |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To count the TER edits of a translation against one reference:

    1. Line up the two with the edit distance, as in *How far apart are two words?*.
    2. Try each shift of a block of up to 10 words that holds a word that does not line up, to a place where the reference has the same block, and work out the edit distance after it.
    3. Make the shift that lowers the edit distance most; of equal gains, the longest block. Go back to step 1 with the shifted translation, until no shift lowers the edit distance.
    4. The TER edits are the shifts plus the edit distance that is left.

    With more than one reference, take the reference that needs the fewest edits, and divide by the average length of the references. For many sentences, add up the edits and the average lengths of all sentences first, and divide once, as BLEU did at the corpus level.

    This lesson's TER works as sacreBLEU does: blocks of up to 10 words, at most 50 places from the same block in the reference, and at most 1,000 shifts tried. On the words of the lesson, small letters without punctuation as in *Translating by rules*, it gives the same edits as sacreBLEU for every sentence.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Take the example of Snover and colleagues, with letters for words: the hypothesis *a d e b c f* and the reference *a b c d e f c*. "The words “b c” in the hypothesis can be shifted to the left". `C.distance` gives the edit distance of two lists of words, and `C.ter_edits` the TER edits:
    """)
    return


@app.cell
def example_cell():
    _hypothesis = ["a", "d", "e", "b", "c", "f"]
    _reference = ["a", "b", "c", "d", "e", "f", "c"]
    print(C.distance(_hypothesis, _reference))
    print(C.ter_edits(_hypothesis, _reference))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The edit distance gives 4. It can line up *b c* or *d e* with the reference, but not both, so the other block costs words that are typed in one place and deleted or substituted in another. TER gives 2: one shift moves *b c* to place 1, which makes *a b c d e f*, and one insertion adds the last *c*. With 7 words in the reference, TER is 2/7 × 100, about 28.6, and without shifts 4/7 × 100, about 57.1. A translator who sees the hypothesis does the same: drag *b c* to the front, and type one word.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shift():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. shift: move a block of words

        Write `shift`, which gets `words`, a list of words, and moves the block of `length` words that starts at place `start` to place `to`. Places count from 0, and `to` counts in the list without the block: with `to` 0, the block goes to the front. In the start code, `block` is the block, `rest` the list without it, and the `return` puts the block between `rest[:to]`, the first `to` words of `rest`, and `rest[to:]`, the others: `+` puts two lists one after the other. Replace the two `...`.

        The check calls `shift(["a", "d", "e", "b", "c", "f"], 3, 2, 1)`, which gives `["a", "b", "c", "d", "e", "f"]`: the shift of Snover and colleagues, with `["b", "c"]` as the block and `["a", "d", "e", "f"]` as the rest. Then `shift(["can", "i", "this", "eat"], 2, 1, 3)`, which gives `["can", "i", "eat", "this"]`: the word-for-word translation of *Kan ik dit eten?*, with *this* moved to the end. And cases of its own: a block at the front or at the end, a block that stays where it is, and word-for-word translations with the verb at the end, as Dutch puts it.
        """),
        L.task("Move a block of words in a list to another place."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shift():
    box_shift = L.code_box("shift", lines=4)
    box_shift
    return (box_shift,)


@app.cell(hide_code=True)
def check_shift(box_shift):
    result_shift = L.check("shift", box_shift.value)
    result_shift
    return (result_shift,)


@app.cell(hide_code=True)
def hints_shift():
    L.hints(
        "shift",
        [
            "A slice `words[a:b]` gives the words from place `a` up to, but not including, place `b`, as in *Lists*. `block` is the slice of `length` words that starts at place `start`. `rest` is the words before the block followed by the words after it, put together with `+`.",
            "The block stops at place `start + length`, so the words after it are `words[start + length:]`, and the words before it `words[:start]`. A slice needs a place after the colon, and `length` is a number of words, not a place.",
            "`    block = ___`",
        ],
        why="`block = words[start:start + length]` and `rest = words[:start] + words[start + length:]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_shift():
    mo.md(r"""
    Once your `shift` runs, the picture under the check works out TER with it on all 3,678 sentences, for three ways to translate: word for word as in *Translating by rules*, Qwen3-0.6B, and one translation by people against the others, as in *Scoring a translation: BLEU and chrF*. Shifts matter most for the word-for-word translations, which keep the Dutch word order: *Mijn vader heeft een nieuwe auto gekocht.* becomes *my father has a new car bought*, which needs one shift of *bought* and nothing more. 1,355 of the 3,678 word-for-word translations need at least one shift, and TER drops from 47.7 without shifts to 43.3. Qwen's translations need a shift in only 343 sentences, and score 36.9 without shifts and 36.2 with them; the translations by people 25.9.

    Qwen is not better on every sentence. Its translation needs fewer TER edits than the word-for-word one for 1,586 sentences, more for 876, and as many for 1,216. On the 1,772 sentences of up to four Dutch words, the word-for-word translations score better, 33.7 against 39.8; on the 1,906 longer ones Qwen wins, 34.3 against 48.3. And Qwen does not always translate: 51 times it gives back the Dutch sentence, as with *Echt?*, which is right only twice, as with *Stop!*. 23 times it gives the last sentence of the prompt, *Reply with only the English translation.*, as for *Vergeet ons niet!*. And 5 times it repeats *The answer is:* until it stops at 64 tokens.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fuzzy():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. fuzzy: how close is a stored sentence?

        Write `fuzzy`, which gets `new`, the words of a new sentence, and `stored`, the words of a sentence in the memory, both lists, and gives back their fuzzy match as a number from 0 to 1. The course's `distance` gives the edit distance of two lists of words, as your `edit_distance` of *How far apart are two words? Edit distance* did: `distance(["ik", "ben", "moe"], ["ik", "ben", "blij"])` gives 1. `max` gives the largest of the numbers it gets: `max(3, 5)` is 5. Replace the two `...`: `edits`, and `longest`, the length of the longer list.

        The check calls `fuzzy(["ik", "heb", "geen", "tijd", "om", "te", "lezen"], ["ik", "heb", "geen", "tijd", "om", "te", "schrijven"])`, which gives about 0.8571: one substitution on 7 words. Then `fuzzy(["ik", "weet", "niet", "wat", "dat", "is"], ["ik", "weet", "niet", "zeker", "wat", "dat", "is"])`, which gives about 0.8571 too: one insertion, and the longer list has 7 words. And cases of its own: two sentences that are the same, two without a word in common, a longer new sentence, and sentences of the memory.
        """),
        L.task("Score how close a stored sentence is to a new sentence, from their edit distance."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fuzzy():
    box_fuzzy = L.code_box("fuzzy", lines=4)
    box_fuzzy
    return (box_fuzzy,)


@app.cell(hide_code=True)
def check_fuzzy(box_fuzzy):
    result_fuzzy = L.check("fuzzy", box_fuzzy.value)
    result_fuzzy
    return (result_fuzzy,)


@app.cell(hide_code=True)
def hints_fuzzy():
    L.hints(
        "fuzzy",
        [
            "`edits` is the edit distance of `new` and `stored`, from the course's `distance`. `longest` is the length of the longer of the two lists: `len` gives the length of a list, as in *Lists*, and `max` the larger of two numbers.",
            "Divide by the longer length, not by the length of `new` or of `stored` alone: two sentences need at most as many edits as the longer one has words, and only that length keeps the score from 0 to 1. `max` gets the two lengths, not the two lists.",
            "`    longest = ___`",
        ],
        why="`edits = distance(new, stored)` and `longest = max(len(new), len(stored))`.",
    )
    return


@app.cell(hide_code=True)
def text_after_fuzzy():
    mo.md(r"""
    The picture under the check compares two starting points for a post-editor: the translation of the best match in the memory, unchanged, and Qwen's translation. Over all 3,678 sentences, the memory scores worse, TER 42.5 against 36.2, but that depends on the match. 113 sentences have a 100% match: the same words, with other punctuation, such as *Wees niet bang!* for *Wees niet bang.*. Their TER is 16.7, and 85 of the 113 need no edit at all. The other 28 do: the stored translation can say it in other words than the references, such as *It can't be!* for *Dat kan niet waar zijn.*, and the punctuation that the match leaves out can change the meaning: *Tom wacht.* finds *Tom, wacht!*, with *Tom, wait!*. From 85% to 99%, the memory still wins, 25.7 against 29.2. From 70% to 84%, the two come close, 35.5 against 33.8, and below 70% Qwen wins by far: 55.0 and 74.6 against 40.7 and 42.8. Most sentences fall in those middle bands: 2,015 from 70% to 84%, and 1,308 from 50% to 69%.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_match():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. best_match: look up the best match

        Write `best_match`, which gets `new`, the words of a new sentence, and `memory`, a list of pairs: the words of a stored sentence, and its English translation. Give back the best fuzzy match and its translation, as a pair. Of equal matches, take the one that comes first in the memory, and when no stored sentence scores above 0, give back `(0, None)`: `None` is Python's value for nothing. In the start code, `best_score` starts at 0 and `best` at `None`, and the loop takes one pair at a time, with `score` its fuzzy match from the course's `fuzzy`, which works as yours does. Replace the two `...`: the `if`, and what `best` keeps.

        The check calls `best_match(["ik", "ben", "moe"], [(["ik", "ben", "blij"], "I'm happy."), (["ik", "ben", "erg", "moe"], "I'm very tired."), (["jij", "bent", "moe"], "You're tired.")])`, which gives `(0.75, "I'm very tired.")`. Then `best_match(["ik", "heb", "geen", "tijd", "om", "te", "lezen"], [(["ik", "heb", "geen", "tijd"], "I have no time."), (["ik", "heb", "geen", "tijd", "om", "te", "schrijven"], "I don't have time to write.")])`, which gives a score of about 0.8571 and *I don't have time to write.* And cases of its own: two equal matches, an empty memory, a memory without a match, and small memories of 20 sentences from the course's memory, which `C.near` gives.
        """),
        L.task("Find the stored sentence with the best fuzzy match, and give back its score and its translation."),
    ])
    return


@app.cell(hide_code=True)
def exercise_match():
    box_match = L.code_box("best_match", lines=9)
    box_match
    return (box_match,)


@app.cell(hide_code=True)
def check_match(box_match):
    result_match = L.check("best_match", box_match.value)
    result_match
    return (result_match,)


@app.cell(hide_code=True)
def hints_match():
    L.hints(
        "best_match",
        [
            "The `if` asks whether `score` is higher than `best_score`, the best score so far. Under it, `best_score` takes the new score, and `best` keeps the translation of the same pair.",
            "Ask for higher, not for higher or equal: with `>=`, a later stored sentence with the same score takes the place of the first. And keep `translation`, not `stored`: a translator wants the English, not the Dutch words that matched.",
            "`        if ___:`",
        ],
        why="In the `if`: `score > best_score`; under it: `best = translation`.",
    )
    return


@app.cell(hide_code=True)
def text_after_match():
    mo.md(r"""
    The picture under the check shows what a fuzzy match can and cannot do. *Wees niet bang.* finds *Wees niet bang!*, and its translation *Don't be afraid!* needs no edit. *Ik heb geen tijd om te lezen.* finds *Ik heb geen tijd om te schrijven.*, at 85.7%, with *I don't have time to write.*: one edit, but without it the translation says something else. TER counts edits, not meaning: Qwen's *I have no time to read.* is right, and needs 2 edits against the translations by people. *Ik wist niet dat je zou komen.* finds *Ik wist wel dat je zou komen.*, also at 85.7%, with *I knew you'd come.*: one word turns the meaning into its opposite, and Qwen wrote *I knew that you would come.*, the same mistake. *Vergeet ons niet!* finds *Vergeet mij niet!*, at 66.7%: *Don't forget about me!* needs one edit, where Qwen gave a sentence of the prompt. And *Als het nodig is, kom ik snel.* finds *Als het mogelijk is, zou ik een uitleg willen.*, at 44.4%, and *If possible, I would like an explanation.* helps little.

    A fuzzy match compares the words of the source, not the meaning, so a translator reads every match with care.

    ### Try it: count your own post-edit

    Type a machine translation and your post-edit of it, and see the shifts and the word edits of TER against your post-edit: the edits that HTER counts. With no other references here, the box divides by the length of your post-edit. The boxes start with Qwen's translation of *Ik hoop je snel te zien.*, *I hope you soon see.*, and a post-edit of it, *I hope to see you soon.*: one shift of *see* and one insertion of *to*, while the edit distance without shifts needs 3.
    """)
    return


@app.cell(hide_code=True)
def box_texts():
    try_translation = mo.ui.text(value="I hope you soon see.", label="The machine translation", full_width=True)
    try_post_edit = mo.ui.text(value="I hope to see you soon.", label="Your post-edit", full_width=True)
    mo.vstack([try_translation, try_post_edit])
    return try_post_edit, try_translation


@app.cell(hide_code=True)
def box_cell(try_post_edit, try_translation):
    mo.Html(C.box_picture(L.lang, try_translation.value, try_post_edit.value))
    return


@app.cell(hide_code=True)
def text_exercise_targeted():
    mo.md(r"""
    ### Exercise 4. Why is HTER lower?

    The course made targeted references of its own. It post-edited Qwen's translations of the first 40 sentences of this lesson that Qwen did not translate as people did, and changed no more than fluent English with the meaning of the Dutch sentence needs. 8 of the 40 needed no change, such as *Can you show me another one?* for *Kunt u me nog een andere laten zien?*: HTER 0, while TER counts 2 edits against *Please show me another one.* and *Please show me another.*, on 4.5 words on average. Over the 40, HTER is 62.4 and TER 74.5: 123 edits against 147, both divided by the average lengths of the translations by people. Only for *Ik ga.* is it the other way round: Qwen's *Go.* became *I'm going.*, 3 edits, while the translation *I go.* by people needs 1, as TER takes the closest of all the references and HTER has one. One translation weighs heavily: *The answer is:*, repeated until 64 tokens, for *Het kan niet waar zijn.*, which takes 48 edits either way; without it, HTER is 39.0 and TER 51.5. Snover and colleagues, with four translations by people per sentence, found: "HTER (i.e., TER with one human-targeted reference) reduces the edit rate by 33% relative to TER with 4 untargeted references".
    """)
    return


@app.cell(hide_code=True)
def choice_targeted_cell():
    targeted = L.choice(
        "targeted",
        question="Why does TER against translations by people count edits that a post-editor does not make?",
        options={
            "mistakes": "People make mistakes when they translate",
            "right": "People say the same thing in other words",
            "shifts": "TER counts shifts, and HTER does not",
            "longer": "Translations by people are longer, so they need more edits",
        },
        correct="right",
        why={
            "right": "Each translation by people was made without seeing Qwen's, so it says the same thing in its own words, and TER counts every such difference as an edit. A post-editor starts from Qwen's translation and changes only what is wrong: *Can you show me another one?* stays as it is, where TER counts 2 edits against *Please show me another one.*",
            "mistakes": "A poor translation by people changes little: TER takes the reference that needs the fewest edits. What do the translations by people not have in common with Qwen's translation, that a post-edit does have?",
            "shifts": "HTER is TER against a targeted reference, with shifts like any TER. What does a targeted reference keep that a translation by people does not?",
            "longer": "*Please show me another one.* has 5 words, and Qwen's translation 6: length is not it. What else differs between the two?",
        },
    )
    targeted
    return (targeted,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The edit distance of [How far apart are two words? Edit distance](/course/open/nlp-edit-distance?lang=en) counted the edits of the word error rate; TER adds shifts of blocks of words, and the same distance gave the fuzzy matches of a translation memory. [Scoring a translation: BLEU and chrF](/course/open/nlp-bleu?lang=en) counted what a translation shares with translations by people; TER counts what a translator still has to change, and HTER only the edits to the post-edit. The next lesson, *Translating with an LLM*, translates with a large language model, as Qwen did here, and looks at what a prompt can change.

    **An everyday example.** A teacher who corrects an essay with a red pen marks what this essay needs, not every way in which it differs from a model essay: two good essays can differ in every sentence.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_shift, result_fuzzy, result_match, targeted):
    L.wrap_up(result_shift, result_fuzzy, result_match, targeted)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
