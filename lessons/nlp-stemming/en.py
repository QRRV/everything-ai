# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-stemming. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-stemming", "en")
    C = L.checks   # the course's words, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [Words and their parts](/course/open/nlp-morphemes?lang=en), and one about [Normalizing: case, accents and stop words](/course/open/nlp-normalize?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_inflection():
    quick_inflection = L.choice(
        "quick-inflection",
        question="Which pair is two forms of one word, made by inflection?",
        options={
            "derive": "*happy* and *happiness*",
            "right": "*walk* and *walked*",
            "compound": "*dog* and *doghouse*",
        },
        correct="right",
        why={
            "right": "*-ed* makes the past of the same verb: inflection. A dictionary has one entry for *walk* and *walked*, and this lesson brings such forms together.",
            "derive": "*-ness* makes a new word, a noun from an adjective: that is derivation. See *Three ways to make words* in [Words and their parts](/course/open/nlp-morphemes?lang=en).",
            "compound": "*doghouse* joins two words into one: a compound. See *Three ways to make words* in [Words and their parts](/course/open/nlp-morphemes?lang=en).",
        },
    )
    quick_inflection
    return (quick_inflection,)


@app.cell(hide_code=True)
def quick_check_fold():
    quick_fold = L.choice(
        "quick-fold",
        question="Fold, from *Normalizing: case, accents and stop words*, makes a word small and takes its accents off. Which pair does it make into one string?",
        options={
            "right": "*één* (one) and *een* (a)",
            "plural": "*Cafés* and *cafe*",
            "niet": "*niet* (not) and *niets* (nothing)",
        },
        correct="right",
        why={
            "right": "Without its accents, *één*, the number one, is *een*, the article a. Fold only changes letters, and it cuts nothing off a word: this lesson does that.",
            "plural": "Fold makes *Cafés* into *cafes*: small letters and no accent. But the *s* of the plural stays, so *cafes* and *cafe* are still two strings. See *Accents* in [Normalizing: case, accents and stop words](/course/open/nlp-normalize?lang=en).",
            "niet": "Fold changes letters, not words: *niet* and *niets* stay two strings. See *Accents* in [Normalizing: case, accents and stop words](/course/open/nlp-normalize?lang=en).",
        },
    )
    quick_fold
    return (quick_fold,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A word comes in many forms. In English, *walk*, *walks*, *walked* and *walking*; in Dutch, *lopen*, *loopt*, *liep* and *gelopen*. To a program that compares strings, those are four different words: a search for *walking* misses a text that says *walked*, and a program that counts words counts one word as four. Christopher Manning, Prabhakar Raghavan and Hinrich Schütze write in the chapter *The term vocabulary and postings lists* of *Introduction to Information Retrieval*: "For grammatical reasons, documents are going to use different forms of a word, such as organize, organizes, and organizing." This lesson brings the forms of a word together in two ways. A **stemmer** chops endings off by rules, and a **lemmatizer** looks up the form that a dictionary uses. Both turn many words into fewer, and each makes mistakes of its own: Martin Porter's stemmer of 1980 gives *university* and *universe* the same stem, and leaves *ran* apart from *run*.

    **An everyday example.** A web shop that shows *running shoes* when you search for *run shoe* has brought *running* and *run*, and *shoes* and *shoe*, together.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="UD_English-EWT, a collection of English sentences from the web, holds 12,415 different word forms, leaving out names, numbers and punctuation. Porter's stemmer gives each form a stem. How many different stems come out?",
        options={
            "few": "About 4,000",
            "right": "8,206",
            "most": "About 11,000",
            "all": "Nearly 12,000",
        },
        reveal="**8,206**: 33.9% fewer than the 12,415 forms. Porter found about the same in 1980, on a list of 10,000 words, and wrote in *An algorithm for suffix stripping*: \"Thus the suffix stripping process reduced the size of the vocabulary by about one third.\" Grouped by the form a dictionary uses, which EWT writes down for every word, the same forms make 8,745 groups. So the stems are even fewer than the lemmas: the stemmer also brings together forms of different words, as the sections below show.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### One word, many forms

    [Words and their parts](/course/open/nlp-morphemes?lang=en) cut words into morphemes. Inflection makes the forms of one word, such as *walked* from *walk*; derivation makes a new word, such as *careful* from *care*. Manning and colleagues: "The goal of both stemming and lemmatization is to reduce inflectional forms and sometimes derivationally related forms of a word to a common base form." Their examples: *am*, *are* and *is* become *be*, and *car*, *cars*, *car’s* and *cars’* become *car*.

    The lesson counts words in two **treebanks**: collections of sentences in which every word comes with what people wrote down about it, among other things the form a dictionary uses. Both come from Universal Dependencies (UD), a project that collects treebanks of many languages in one format, release 2.19 (November 2026). *UD_English-EWT* has sentences from the web: blogs, newsgroups, reviews and questions with their answers; the course left out its emails, which hold real names and addresses. *UD_Dutch-Alpino* has sentences from Dutch newspapers and other written Dutch. The lesson counts every word in small letters, except names, numbers, punctuation, symbols and the words that the treebank tags as other: 12,415 different forms in EWT and 19,982 in Alpino. The lesson *Corpora and treebanks* tells more about treebanks.

    ### Stemming: chopping by rules

    Stemming, Manning and colleagues write, "usually refers to a crude heuristic process that chops off the ends of words in the hope of achieving this goal correctly most of the time, and often includes the removal of derivational affixes." A heuristic is a rule of thumb: it works often, not always. What a stemmer leaves of a word is its **stem**. "The most common algorithm for stemming English, and one that has repeatedly been shown to be empirically very effective, is Porter’s algorithm (Porter 1980)," they go on. Martin Porter wrote it in 1979 at the Computer Laboratory in Cambridge, England, as part of a larger project in information retrieval, IR: the field of search engines. In *An algorithm for suffix stripping*, Porter says what the rules are for: "the suffixes are being removed simply to improve IR performance, and not as a linguistic exercise."

    Porter's stemmer goes through five steps, and each step has groups of rules. A rule reads (condition) S1 → S2, which Porter explains: "This means that if a word ends with the suffix S1, and the stem before S1 satisfies the given condition, S1 is replaced by S2." The first group, step 1a, has no conditions:

    | rule | example |
    |---|---|
    | SSES → SS | *caresses* → *caress* |
    | IES → I | *ponies* → *poni* |
    | SS → SS | *caress* → *caress* |
    | S → | *cats* → *cat* |

    *caresses* ends with *sses*, but also with *ss* and with *s*. Porter: "In a set of rules written beneath each other, only one is obeyed, and this will be the one with the longest matching S1 for the given word." So *caresses* takes the first rule. The rule SS → SS changes nothing: it is there so that *caress* takes it, and not the rule S →, which would make *caress* into *cares*. Later steps take long endings off bit by bit: "Thus GENERALIZATIONS is stripped to GENERALIZATION (Step 1), then to GENERALIZE (Step 2), then to GENERAL (Step 3), and then to GENER (Step 4)."

    A stem need not be a word: *ponies* becomes *poni*. People often take that for an error, says Porter's page about the stemmer, and it answers: "But the purpose of stemming is to bring variant forms of a word together, not to map a word onto its ‘paradigm’ form." Manning and colleagues: "However, the exact stemmed form does not matter, only the equivalence classes it forms": what counts is which words end up with the same stem.

    ### The measure m

    Most later rules have a condition on what is left before the suffix. Porter: "A *consonant* in a word is a letter other than A, E, I, O or U, and other than Y preceded by a consonant." Every other letter is a vowel: "So in TOY the consonants are T and Y, and in SYZYGY they are S, Z and G." The **measure** m of a word, or of a part of a word, counts how often a vowel is followed by a consonant. Porter's examples: "m=0 TR, EE, TREE, Y, BY." "m=1 TROUBLE, OATS, TREES, IVY." "m=2 TROUBLES, PRIVATE, OATEN, ORRERY." In *trouble*, only the *ou* is followed by a consonant, the *b*. In *ivy*, the *y* follows a consonant, so it is a vowel, and only the *i* is followed by a consonant.

    A rule with the condition (m > 1) fires only when what would be left has an m above 1. Porter's example is the rule (m > 1) EMENT →, which takes *ement* off: "This would map REPLACEMENT to REPLAC, since REPLAC is a word part for which m = 2." Manning and colleagues add that the rule "would map replacement to replac, but not cement to c": *c* has m = 0. The measure "loosely checks the number of syllables to see whether a word is long enough that it is reasonable to regard the matching portion of a rule as a suffix rather than as part of the stem of a word," they write. And Porter: "There is no linguistic basis for this approach."

    ### Lemmatization: looking it up

    Lemmatization, Manning and colleagues write, "usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma." Daniel Jurafsky and James Martin give an example in the chapter *Embeddings* of *Speech and Language Processing*: "The form mouse would also be the lemma for the word mice; dictionaries don’t have separate definitions for inflected forms like mice." And: "Similarly sing is the lemma for sing, sang, sung." A program that finds lemmas is a **lemmatizer**. It puts together what no rule for endings can: *ran* and *run*, *went* and *go*.

    Both treebanks have a column LEMMA. The guidelines of UD: "The LEMMA field should contain the canonical or base form of the word, which is the form typically found in dictionaries." And: "The lemma does not remove derivational morphology, so the lemma of [en] organizations is organization not organize (nor organ)." Porter's stemmer makes *organization* into *organ*. In Dutch, the lemma of a verb is the infinitive, such as *lopen* for *liep*. EWT's page says that its lemmas were "assigned by a program, with some manual corrections, but not a full manual verification", so some of them may be wrong.

    A lemma can depend on the sentence. Manning and colleagues: "If confronted with the token saw, stemming might return just s, whereas lemmatization would attempt to return either see or saw, depending on whether the use of the token was as a verb or a noun." Porter's stemmer leaves *saw* as it is. In EWT, *saw* has the lemma *see* 25 times and *saw* once, and *best* has the lemma *good* 246 times and *well* 24 times. In Alpino, *liep* has the lemma *lopen* 17 times and *op_lopen* 7 times: Alpino joins the separate part of a separable verb, such as the *op* of *liep op*, to the lemma.

    ### Too much and too little

    Wessel Kraaij and Renée Pohlmann, who wrote a Dutch version of Porter's stemmer, name the two kinds of mistake in *Porter's stemming algorithm for Dutch* (1994): "'Over-stemming errors' are those errors which result in the conflation of semantically unrelated words." And: "The term 'under-stemming errors' is used for those errors where a failure to conflate semantically related words is concerned." To conflate is to bring together. **Overstemming**: Porter's stemmer gives *university* and *universe* the stem *univers*, and *general* and *generous* the stem *gener*; in an exercise, Manning and colleagues list *university* and *universe* among the pairs that Porter's stemmer puts together, and ask which should not be. Porter saw it coming: "For example, if SAND and SANDER get conflated, so most probably will WAND and WANDER." **Understemming**: *ran* keeps the stem *ran*, and *went* the stem *went*. Porter chose not to handle words whose stem changes, such as *deceive* and *deception*: "In view of the error rate that must in any case be expected, it did not seem worthwhile to try and cope with these cases." And mending one kind of mistake brings the other, Kraaij and Pohlmann write: "If suffix stripping rules are added or modified in order to reduce the understemming errors, these modifications will likely introduce more overstemming errors."

    For a search engine, a stemmer finds more of the texts it should find, and also more texts it should not. Manning and colleagues' example: Porter's stemmer gives *operate*, *operating*, *operational* and four more forms the stem *oper*, so a search for *operating system* also finds texts with *operative* and *system* in them. For English, they write, "either form of normalization tends not to improve English information retrieval performance in aggregate – at least not by very much. Although it helps a lot for some queries, it equally hurts performance a lot for others." And: "The situation is different for languages with much more morphology (such as Spanish, German, and Finnish). Results in the European CLEF evaluations have repeatedly shown quite large gains from the use of stemmers".

    ### Dutch

    A Dutch verb has many forms: *lopen*, *loop*, *loopt*, *liep*, *liepen*, *lopend* and *gelopen*. The spelling makes it harder for rules, Kraaij and Pohlmann note: "Long vowels in Dutch are spelled single in open syllables and double in closed ones (e.g. schaap - schapen)." So *loopt* has *oo*, and *lopen* one *o*. The course uses Porter's own stemmer for Dutch, as the Snowball project describes it, not the one of Kraaij and Pohlmann. It takes off endings such as *-en*, *-e*, *-heid*, *-ing* and *-lijk*, and makes a double vowel before a final consonant single, as in *maan* → *man*. That gives *lopen* the stem *lop*, but *loopt* stays *loopt*, *gelopen* becomes *gelop*, and *liep* stays *liep*: four stems for one verb.

    Since version 3.0.0 (2025), Snowball's default for Dutch is the stemmer of Kraaij and Pohlmann: "Based on feedback from Dutch users, in Snowball 3.0.0 we switched the default Dutch stemmer to use the Kraaij-Pohlmann algorithm." It gives *lopen*, *loopt* and *gelopen* the stem *loop*, and it goes further; Snowball's page on it says: "Both prefixed and infixed ge are removed, so for example geluidgevoelige stems to luidvoel." The course uses Porter's stemmer because its rules are written out step by step; the stemmer of Kraaij and Pohlmann "is effectively only specified by its implementation in ANSI C," Snowball writes. Neither stemmer puts *liep* with *lopen*. Kraaij and Pohlmann, about verbs such as *drinken*, *dronk*, *gedronken*: "For obvious reasons, a simple suffix-stripper will never be able to map the different forms of these types of verbs onto a single stem." Dutch school grammar has a *stam* of a verb too, the infinitive without its ending as you say it, so the stam of *lopen* is *loop*; a stem in this lesson is whatever a stemmer leaves, such as *lop*.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **stemming** | chopping the endings off words by rules, so that the forms of a word come out the same |
        | **stemmer** | a program that does stemming, such as Porter's stemmer |
        | **stem** | what a stemmer leaves of a word, such as *poni* for *ponies*; it need not be a word |
        | **lemma** | the form of a word that a dictionary uses: *run* for *ran*, *mouse* for *mice*, *lopen* for *liep* |
        | **lemmatization**, **lemmatizer** | finding the lemma of each word, with a dictionary or a table; a program that does it |
        | **overstemming**, **understemming** | a stemmer bringing together forms of different words, such as *university* and *universe*; a stemmer leaving forms of one word apart, such as *ran* and *run* |
        | **measure** m | Porter's count of how often a vowel is followed by a consonant: 0 for *tree*, 1 for *trouble*, 2 for *troubles* |
        | **treebank** | a collection of sentences in which every word comes with what people wrote down about it, such as its lemma |
        | `word.endswith(suffix)` | `True` when the string `word` ends with the string `suffix`: `"cats".endswith("s")` is `True` |
        | `for suffix in rules` | a `for` over a dictionary goes over its keys, in the order they were put in |
        | `a and not b`, `a or b` | `not` turns `True` into `False` and back; `or` is `True` when at least one side is `True` |
        """),
        L.watch_out("In mathematics, a *lemma* is something else: a small result on the way to a bigger one. In this lesson, a lemma is the form of a word that a dictionary uses. And *dictionary* means two things here: a book of words, and a Python dictionary, such as `TABLE`, with keys and values."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Ask what the task needs: only groups of forms that belong together, or words that a person reads?
    2. For groups: give every word its stem, with a stemmer for its language, and compare the stems.
    3. For the dictionary form: look every word up in a table of lemmas; a word that is not in it stays as it is.
    4. Count what the step brings together that does not belong together, and what it leaves apart.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell takes *ponies* through step 1a by hand: which suffixes it ends with, how long the word and the suffix are, and what is left with the new ending.
    """)
    return


@app.cell
def example_ponies():
    W = "ponies"
    print(W.endswith("sses"), W.endswith("ies"), W.endswith("s"))
    print(len(W), len("ies"))
    print(W[:3] + "i")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *ponies* ends with *ies* and with *s*, and the longer suffix decides: the rule IES → I. Of the six letters, the three before the suffix stay, and *i* comes after them: *poni*. Exercise 1 does this for any word and any group of rules.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rules():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. apply_rules: one group of rules

        `STEP_1A` below holds step 1a as a dictionary: each suffix with its new ending, the longest suffix first, so that the first rule that fits is the one Porter takes. Write `apply_rules(word, rules)`: go over the suffixes of `rules` in their order, and for the first suffix that `word` ends with, give back the word without that suffix, with the rule's new ending glued to it. When no suffix fits, give back `word` as it is. For `apply_rules("caresses", STEP_1A)`, the result is `"caress"`. Replace the `...`. The check calls that, then cases of its own: the other rules of step 1a, a word that no rule fits with an *s* inside it, a word that starts with an *s*, two rules of step 2 where the longer suffix must come first, Dutch rules, no rules at all, and an empty word.
        """),
        L.task("Apply the first rule of a group whose suffix ends the word: cut the suffix off and glue the new ending on."),
    ])
    return


@app.cell
def step_1a_cell():
    STEP_1A = {"sses": "ss", "ies": "i", "ss": "ss", "s": ""}
    print(list(STEP_1A), STEP_1A["ies"])
    return


@app.cell(hide_code=True)
def exercise_rules():
    box_rules = L.code_box("apply-rules", lines=5)
    box_rules
    return (box_rules,)


@app.cell(hide_code=True)
def check_rules(box_rules):
    result_rules = L.check("apply-rules", box_rules.value)
    result_rules
    return (result_rules,)


@app.cell(hide_code=True)
def hints_rules():
    L.hints(
        "apply-rules",
        [
            "Go over the suffixes with `for suffix in rules`, and ask `word.endswith(suffix)`. For the first suffix that fits, give back the part of `word` before it with `rules[suffix]` glued on; after the loop, give back `word`.",
            "For *ponies* and *ies*, the first three letters stay. How do you get that three from the word and the suffix, for any word?",
            r'`            return word[:___] + rules[suffix]`',
        ],
        why=r'`for suffix in rules:`, `if word.endswith(suffix):`, `return word[:len(word) - len(suffix)] + rules[suffix]`, and after the loop `return word`.',
    )
    return


@app.cell(hide_code=True)
def text_after_rules():
    mo.md(r"""
    With your apply_rules and `STEP_1A`, 2,563 of the 12,415 forms of EWT change, and 10,854 groups are left. Forms such as *time* and *times* come together, and the treebank agrees. But step 1a also cuts the *s* off words that are not plurals: *is* becomes *i*, *as* becomes *a*, and *its* comes together with *it*. Porter's own program, the page about the stemmer says, leaves words of one or two letters alone, because otherwise "strings of length 2 might lose a final s, so as goes to a and is to i." The course's stemmer does the same. And step 1a is only the first step: *running* is still apart from *run*, until step 1b takes *ing* off.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_measure():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. measure: Porter's m

        `vowel(word, i)` below gives `True` when the letter at place `i` of `word` is a vowel for Porter: *a*, *e*, *i*, *o* or *u*, or a *y* after a consonant. Its loop goes over the letters up to place `i`, and remembers in `is_vowel` whether the letter before was a vowel. Write `measure(word)`: count the places in the word where a vowel is followed by a consonant, as `vowel` sees them, and give back that number. For `measure("trouble")`, the result is `1`. Replace the `...`. The check calls that, then cases of its own: the other examples of Porter's paper, a *y* after a vowel and after a consonant, *crepuscular*, for which Snowball gives m = 4, a short word, and an empty word.
        """),
        L.task("Count how often a vowel is followed by a consonant, with Porter's vowels."),
    ])
    return


@app.cell
def vowel_cell():
    def vowel(word, i):
        is_vowel = False
        for j in range(i + 1):
            is_vowel = word[j] in "aeiou" or (word[j] == "y" and j > 0 and not is_vowel)
        return is_vowel


    print([vowel("toy", i) for i in range(3)])
    print([vowel("syzygy", i) for i in range(6)])
    return


@app.cell(hide_code=True)
def exercise_measure():
    box_measure = L.code_box("measure", lines=6)
    box_measure
    return (box_measure,)


@app.cell(hide_code=True)
def check_measure(box_measure):
    result_measure = L.check("measure", box_measure.value)
    result_measure
    return (result_measure,)


@app.cell(hide_code=True)
def hints_measure():
    L.hints(
        "measure",
        [
            "Count with `m`: start at `m = 0`, go over the places with `for i in range(1, len(word))`, and add one with `m = m + 1` where a vowel is followed by a consonant. Give back `m` after the loop.",
            "The letter at place `i` is the consonant. At which place is the vowel before it, and how do you ask `vowel` about that place?",
            r'`        if ___ and not vowel(word, i):`',
        ],
        why=r'`m = 0`, `for i in range(1, len(word)):`, `if vowel(word, i - 1) and not vowel(word, i):` `m = m + 1`, and after the loop `return m`.',
    )
    return


@app.cell(hide_code=True)
def text_after_measure():
    mo.md(r"""
    With your measure, the rule (m > 1) EMENT → takes *ement* off *replacement* and *management*, where m of what is left is 2. It leaves *statement*, *agreement* and *settlement* alone, where m of what would be left is 1, and *cement* too. Porter's whole stemmer with your measure gives every form of EWT the same stem as the course's, and makes the 8,206 groups of the guess. Porter's page warns about a common mistake in programs that copy the stemmer: in a group of rules, the rule with the longest suffix decides, and "This is true whether the rule succeeds or fails". "Properly, argument stems to argument," the page says: *ment* is the longest suffix that fits, *argu* has m = 1, so *ment* stays, and no shorter suffix gets a turn. A program that went on to try *ent* would make *argum*.
    """)
    return


@app.cell(hide_code=True)
def compare_cell():
    mo.Html(C.compare_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_after_compare():
    mo.md(r"""
    Over the whole of EWT, Porter's stemmer makes 8,206 groups of the 12,415 forms, and the lemmas make 8,745. 1,013 stems bring together forms of more than one lemma, such as *use* and *us*, whose lemma in EWT is *we*, and *business* and *busy*. 515 lemmas have their forms under more than one stem, such as *go*, *going*, *went* and *gone*. In Dutch it goes the other way: Porter's stemmer for Dutch makes 16,561 groups of the 19,982 forms of Alpino, more than the 14,928 lemmas. Fewer of its stems bring forms of more than one lemma together, 899, such as *zaal* (hall) and *zal* (will), and 1,789 lemmas have their forms under more than one stem, such as *werken*, *werkt*, *werkte* and *gewerkt*: it takes *-en* and *-e* off, but leaves the *t* of *werkt* and the *ge-* of *gewerkt*. Porter's page says that the original stemmer for English "should be regarded as ‘frozen’", and adds: "For practical work, therefore, the new Snowball stemmer is recommended." That newer stemmer for English still changes: since Snowball 3.0.0, it no longer puts *university* and *universe* together.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lemmatize():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. lemmatize: look it up

        A treebank comes in parts, so that a program can be made from one part, the training part, and tried on another, the test part. `TABLE` below holds the course's table of lemmas for English: each of the 11,115 different forms in the training part of EWT, with its most common lemma there. `TABLE_NL` holds the same for the 18,542 forms in the training part of Alpino. Write `lemmatize(word, table)`: when `word` is a key of the dictionary `table`, give back its value, the lemma; otherwise give back `word` itself, since many words are their own lemma. For `lemmatize("liep", TABLE_NL)`, the result is `"lopen"`. Replace the `...`. The check calls that, then cases of its own: English and Dutch words of the tables, a word that no table has, a word that is its own lemma, a table of its own with a word that is in it and one that is not, and an empty table.
        """),
        L.task("Look a word up in a table of lemmas, and keep the word itself when the table does not have it."),
    ])
    return


@app.cell
def table_cell():
    TABLE = C.table("en")
    TABLE_NL = C.table("nl")
    print(len(TABLE), len(TABLE_NL))
    print(TABLE["feet"], TABLE["went"], TABLE_NL["steden"])
    return


@app.cell(hide_code=True)
def exercise_lemmatize():
    box_lemmatize = L.code_box("lemmatize", lines=4)
    box_lemmatize
    return (box_lemmatize,)


@app.cell(hide_code=True)
def check_lemmatize(box_lemmatize):
    result_lemmatize = L.check("lemmatize", box_lemmatize.value)
    result_lemmatize
    return (result_lemmatize,)


@app.cell(hide_code=True)
def hints_lemmatize():
    L.hints(
        "lemmatize",
        [
            "Two things you know about dictionaries: `table[word]` gives the value of a key, and asking for a key that is not there stops the program with an error. So first one `if` with a question about `table`, then two ways back: the lemma, or `word`.",
            "Which question about the table tells you whether `table[word]` is safe?",
            r'`    if ___:`',
        ],
        why=r'`if word in table:` gives back `table[word]`; otherwise `return word`.',
    )
    return


@app.cell(hide_code=True)
def text_after_lemmatize():
    mo.md(r"""
    The test part of each treebank holds words that the tables were not made from: 14,801 English and 8,912 Dutch ones. On them, your lemmatize gives the treebank's lemma for 96.6% of the English words and 90.5% of the Dutch ones. 94.7% of the English words are in `TABLE`, but only 88.7% of the Dutch words are in `TABLE_NL`: Dutch writes a compound as one word, and a new compound is in no table. Alpino marks the parts of a compound in its lemmas with `_`: the lemma of *jeugdgezondheidszorg* (youth health care) is *jeugd_gezondheid_zorg*. Of the 1,008 Dutch words that `TABLE_NL` does not have, 484 have such a lemma; of the 784 English words that `TABLE` does not have, 2. For those words, the word itself is never right. In all, the word itself is right for 469 of the 784 in English and for 272 of the 1,008 in Dutch. And a table gives each form one lemma, the most common: *best* gets *good*, also where the treebank says *well*, and *vindt* gets *vinden*, also where Alpino says *plaats_vinden*, for *vindt plaats* (takes place). A lemmatizer that looks at the sentence around a word can tell such cases apart.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: stem and lemma of your own words

    Type some words and choose a language. The picture gives each word its stem, from Porter's stemmer for that language, and the lemmas that the treebank gives it, with how often. Try *ran*, *mice* and *better*, or Dutch verbs such as *ging*, *gegaan* and *zag*.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_text = mo.ui.text(value="Ik loop, Tom loopt, wij liepen, en jullie hebben gelopen.", label="Words", full_width=True)
    demo_lang = mo.ui.dropdown(options={"Dutch": "nl", "English": "en"}, value="Dutch", label="Language")
    mo.vstack([demo_text, demo_lang])
    return demo_lang, demo_text


@app.cell(hide_code=True)
def demo_look(demo_lang, demo_text):
    mo.Html(C.demo_picture(demo_text.value, demo_lang.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    In *Ik loop, Tom loopt, wij liepen, en jullie hebben gelopen.* (I walk, Tom walks, we walked, and you have walked.), *loop* gets the stem *lop*, as *lopen* would, but *loopt* keeps its *t*, *liepen* becomes *liep*, and *gelopen* becomes *gelop*: four stems for the forms of one verb. The treebank gives *loopt*, *liepen* and *gelopen* the lemma *lopen*, and *loop* most often the lemma *loop*: in Alpino, *loop* is more often the noun. And *hebben* becomes *hebb*: the stemmer takes *-en* off, and keeps both *b*'s.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Exercise 4. Stem, lemma, or neither?

    Each bold task asks something of a program that works with words. Put it under what it needs.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_stem = L.sort("stem-or-lemma")
    sort_stem
    return (sort_stem,)


@app.cell(hide_code=True)
def check_sort(sort_stem):
    result_sort = L.check("stem-or-lemma", sort_stem)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Normalizing: case, accents and stop words](/course/open/nlp-normalize?lang=en) made *Café* and *cafe* one word. This lesson went on to the forms of one word, which [Words and their parts](/course/open/nlp-morphemes?lang=en) called inflection. A stemmer chops endings off by rules: you wrote a function that applies a group of Porter's rules, and the measure m that later rules ask about, and saw Porter's stemmer bring *university* and *universe* together and leave *ran* apart from *run*. A lemmatizer looks up the form a dictionary uses: yours used a table from a treebank, and missed the new words that the table did not have. Like case folding, both throw something away: the *-ed* that tells the past, the *-s* that tells more than one. The next lesson of the domain, *How far apart are two words? Edit distance*, compares two words letter by letter, and counts how far one is from the other, such as a typo from the word that was meant.

    **An everyday example.** A library catalog that finds a book on *gardening* when you type *garden* has given both words the same stem.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_lemmatize, result_measure, result_rules, result_sort):
    L.wrap_up(result_rules, result_measure, result_lemmatize, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
