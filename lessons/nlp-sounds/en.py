# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-sounds. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-sounds", "en")
    C = L.checks   # the course's words, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Words and their parts](/course/open/nlp-morphemes?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_plural():
    quick_plural = L.choice(
        "quick-plural",
        question="In *cats*, what is *-s*?",
        options={
            "right": "A morpheme that marks the plural",
            "letter": "Only a letter, with no job of its own",
            "word": "A word of its own, as in a compound",
        },
        correct="right",
        why={
            "right": "*-s* is the smallest part of *cats* with a job in the grammar: it makes the plural, by inflection. This lesson meets a Dutch plural ending, the *-en* of *honden*, which lets you hear a sound that the singular *hond* hides.",
            "letter": "*-s* has a job: *cat* is one, *cats* more than one. A part with a meaning or a job is a morpheme. See *Parts of words* in [Words and their parts](/course/open/nlp-morphemes?lang=en).",
            "word": "*-s* cannot stand on its own as a word; a compound joins whole words, as *doghouse* does. See *Three ways to make words* in [Words and their parts](/course/open/nlp-morphemes?lang=en).",
        },
    )
    quick_plural
    return (quick_plural,)


@app.cell(hide_code=True)
def quick_check_derive():
    quick_derive = L.choice(
        "quick-derive",
        question="What does *-ness* do in *happiness*?",
        options={
            "form": "It makes another form of the same word",
            "join": "It puts two whole words together into one",
            "right": "It makes a new word from an existing one",
        },
        correct="right",
        why={
            "right": "*-ness* makes the noun *happiness* from the adjective *happy*: derivation. Note the spelling: *happy* ends in a y, and *happiness* has an i. Letters and morphemes do not line up one to one, and this lesson shows that letters and sounds do not either.",
            "form": "That is what inflection does, as in *walk* and *walked*. *Happiness* is a new word, a noun, where *happy* is an adjective. See *Three ways to make words* in [Words and their parts](/course/open/nlp-morphemes?lang=en).",
            "join": "*-ness* is no word of its own: it is a suffix. A compound joins whole words, as *doghouse* does. See *Three ways to make words* in [Words and their parts](/course/open/nlp-morphemes?lang=en).",
        },
    )
    quick_derive
    return (quick_derive,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A program that speaks has letters and has to make sounds: a navigation app that reads out a street name, or a screen reader. A program that listens has sounds and has to write letters. Letters are a poor guide to sounds. Jurafsky and Martin, in their chapter on phonetics, call speech sounds phones and a spelling an orthography: "the mapping between the letters of English orthography and phones is relatively opaque; a single letter can represent very different sounds in different contexts." Their example: the letter *c* is a k-sound in *cougar*, and an s-sound in *cell*. So speech technology keeps lists of how words sound. In the words of Jackson Lee and colleagues (2020), who built such lists from a free dictionary: "Nearly all speech technologies depend on explicit mappings between the orthographic forms of words and their pronunciations, represented as a sequence of phones." For a word missing from the list, a model learns to turn letters into sounds: grapheme-to-phoneme conversion. Both need a way to write sounds down, and an idea of which differences between sounds matter in a language.

    **An everyday example.** Your navigation app reads out the name of a Dutch street as if it were English. Or you write *hond* with a d, and say it with a t.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="The linguist Ian Maddieson counted the consonants of hundreds of languages for the World Atlas of Language Structures. English has 24, by general agreement. How many has the language in the middle of the count, with half of the languages below it and half above?",
        options={
            "few": "About 10, fewer than half as many as English",
            "right": "About 21, a few fewer than English",
            "many": "About 45, nearly twice as many as English",
        },
        reveal="**About 21, a few fewer than English**: the median of 562 languages in Maddieson's chapter of the World Atlas (2013); the most common count is 22, and the mean 22.7. The range runs from 6 consonants in Rotokas, a language of Papua New Guinea, to 122 in !Xóõ, a language of Botswana, \"mainly because it has a very large number of different click sounds\". Over the 3,020 sound inventories of the PHOIBLE database (version 2.0, 2019), the course counts a median of 22 consonants, and 130 for !Xóõ: a count depends on how a linguist analyzes a language. Even for English, Maddieson adds, \"some linguists might decide there are one or two more or less than this\".",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Sounds, not letters

    **Phonetics** studies the sounds of speech. Jurafsky and Martin define it as "the study of the speech sounds used in the languages of the world, how they are produced in the human vocal tract, how they are realized acoustically, and how they can be digitized and processed." A speech sound is a **phone**, and phoneticians write it in square brackets: [k].

    Letters and sounds go their own ways, in English more than in many languages: Spanish, Jurafsky and Martin note, is much more transparent. The English Wikipedia counts at least eight pronunciations of *ough* in North American English and nine in British English, as in *tough*, *cough*, *though* and *through*. Dutch has no one-to-one match between sounds and letters either, as the Dutch reference grammar, the *Algemene Nederlandse Spraakkunst* (e-ANS), points out. A well-known joke spells *fish* as *ghoti*: *gh* as in *tough*, *o* as in *women*, *ti* as in *nation*. It is often said to be George Bernard Shaw's, but it does not appear in Shaw's writings; its first confirmed use is in a letter from 1855, the year before Shaw was born.
    """)
    return


@app.cell(hide_code=True)
def text_idea_phonemes():
    mo.md(r"""
    ### Sounds that change a word

    Some differences between sounds change a word, and others do not. The International Phonetic Association, in its principles of 1912: "It is necessary to ascertain what are the distinctive sounds in the language, i. e. those which if confused might conceivably alter the meanings of words." Such a sound is a **phoneme**, written between slashes: /p/. Put /b/ in the place of /p/, and *pat* becomes *bat*. Two words that differ in one sound, in the same place, form a **minimal pair**: Dutch has *pak* and *bak*, and *tak* and *dak*. *Kat* and *kast* are not a minimal pair, since *kast* has one sound more.

    Other differences never change a word. The e-ANS gives a Dutch example: the i of *bier* is longer than the i of *biet*, because an r follows. Both are the phoneme /i/: they are two **allophones** of it, and the sound next to them decides which one you hear. The principles of the Association, in their version of 1949, call phonemes "families of related sounds which count linguistically in a given language, as if they were one."

    **Phonology** studies what a language does with its sounds: which differences count, and the rules that sounds follow. The e-ANS draws the line like this: phonology is about the role of sounds in the system of a language, and phonetics about how sounds are made and heard, and what they are physically. Which differences count depends on the language, as the last part of this section shows.
    """)
    return


@app.cell(hide_code=True)
def text_idea_ipa():
    mo.md(r"""
    ### One symbol per sound

    The International Phonetic Association was founded in 1886 in Paris, by Paul Passy and other linguists and language teachers. In 1888 it drew up an alphabet for the pronunciation of any language: the **IPA**, short for International Phonetic Alphabet. One of its principles of 1912: "Typical single sounds should be represented (as far as possible) by single letters without diacritical marks." The chart of the alphabet was last revised in 2015, and since 2025 it is issued again each year, free to share under CC BY-SA 4.0. On the issue of 2026, the course counts 59 symbols in the table of consonants made with air from the lungs, and 28 vowels. More symbols and marks come on top, for clicks, for example, or the small ʰ for a puff of air.

    Computers long used plainer alphabets. Jurafsky and Martin's chapter writes sounds in the ARPAbet, "a simple phonetic alphabet … that conveniently uses ASCII symbols to represent an American-English subset of the IPA". The CMU Pronouncing Dictionary of Carnegie Mellon University uses a version of it with 39 phonemes, for "about 134,000 wordforms", as Jurafsky and Martin put it. In the dictionary, *pat* is P AE1 T, where the 1 marks the vowel with the stress. CELEX, a lexicon of English, Dutch and German, is not free to share. WikiPron, the work of Lee and colleagues, collects the pronunciations of Wiktionary, the free dictionary that anyone can edit: its version 2.2.0, of July 2026, holds 5,225,170 pronunciations in 348 languages, and its main Dutch list has 58,539 entries. The words of this lesson come from Wiktionary too.

    Inside speech models, sounds turn up as well. Jurafsky and Martin, in their chapter on speech recognition, describe the task as mapping "acoustic waveforms to sequences of graphemes": from sound straight to letters. Yet a model that learns from audio without any labels can find units of its own that seem to match sounds. Of wav2vec 2.0, Alexei Baevski and colleagues (2020) write that "many discrete latents appear to specialize in specific phonetic sounds."
    """)
    return


@app.cell(hide_code=True)
def text_idea_devoicing():
    mo.md(r"""
    ### Sounds that change with their place

    A sound can change with its place in a word. In Dutch, plosives such as *d* and fricatives such as *z*, together the **obstruents**, are in principle voiceless at the end of a syllable, the e-ANS says: **final devoicing**. The Taalportaal, a description of Dutch grammar in English: "In Dutch, voiced obstruents are generally not found in syllable-final position. As a consequence, phonologically voiced obstruents occurring syllable-finally are devoiced." Phonologically voiced means voiced in the system of the language, such as the d of *hond*, which you hear in *honden*. A **voiced** sound is made with vibrating vocal folds, and a **voiceless** one without: put your fingers on your throat and say *zzz*, then *sss*. The picture shows the five pairs in the e-ANS's table of the rule.

    The morpheme *hond*, the e-ANS notes, has two sound forms: [hɔnt] in *hond*, and [hɔnd] in *honden*. In *honden*, the d starts the second syllable, so it stays voiced. The rule works at the end of every syllable, also inside a word: *handrem* has a t-sound in the middle. And in connected speech another rule can overrule it: in *huisdeur* (front door), the e-ANS notes, the s of *huis* can sound as a z again before the voiced d. Dutch, the e-ANS adds, has words like *kat* [kɑt], but none like English *bed* [bɛd]. English keeps *bad* and *bat* apart, though its voiced obstruents at the end of a word are "devoiced to some extent", according to the English Wikipedia. The German of Germany has the rule too: *Rad* (wheel) and *Rat* (advice) sound the same.

    Dutch spelling hides the rule for b, d and g, the letter for ɣ. The *Woordenlijst*, the official word list of Dutch, explains this with the principle of uniformity: a word is spelled the same way as far as possible, so *bloed* (blood) is always written with a d, though you hear a t. But Dutch writes no v or z at the end of a syllable: *raaf* next to *raven*, and *prijs* next to *prijzen*.
    """)
    return


@app.cell(hide_code=True)
def rule_picture():
    mo.Html(C.rule_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **phonetics** | the study of speech sounds: how they are made, how they travel as sound and how they are heard |
        | **phone** | a speech sound, written in square brackets: [t] |
        | **phoneme** | a sound that can change a word in a language, written between slashes: /t/ |
        | **minimal pair** | two words that differ in one sound, in the same place, such as *pat* and *bat* |
        | **allophone** | one of the sounds that count as the same phoneme, such as the two i's of *bier* and *biet* |
        | **phonology** | the study of what a language does with its sounds: which differences count, and the rules sounds follow |
        | **IPA** | the International Phonetic Alphabet: one symbol per sound as far as possible, for any language |
        | **voiced, voiceless** | made with or without vibrating vocal folds: *z* is voiced, *s* voiceless |
        | **obstruent** | a sound made by closing or narrowing the way for the air: a plosive such as *p* or *d*, or a fricative such as *f* or *z* |
        | **final devoicing** | the Dutch rule that makes a voiced obstruent voiceless at the end of a syllable: *hond* ends in a t-sound |
        | **ɔ ɑ ɪ ɛ u ə** | vowels in the IPA: the vowel of *hond*, *kat*, *rib*, *pet* and *hoed*, and the e of the ending in *honden* |
        | **aː, œy̯** | the long aa of *zaag*, where ː marks a long vowel, and the ui of *druif* |
        | **ɣ, x** | the g of *zagen*, and the voiceless g at the end of *zaag*, like the ch of *lach* (laugh) |
        | **æ ʃ θ ŋ** | in English: the a of *pat*, the sh of *ship*, the th of *think* and the n of *think* |
        | `s[len(s) - 1]` | the last value of a list `s`: the places run from 0 to `len(s) - 1` |
        """),
        L.watch_out("Final devoicing is about sound, not spelling: *hond* keeps its d in writing. And it is a rule of Standard Dutch, not of every Dutch dialect: the dialect of the island of Ameland, the English Wikipedia notes, has no final devoicing."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Write each word as a list of its sounds, one IPA symbol per sound, after a pronunciation dictionary such as Wiktionary.
    2. To test whether two sounds are different phonemes, look for a minimal pair: two words of the same length that differ only in those two sounds.
    3. For a rule such as final devoicing, take a form of the word where the sound is not at the end, such as the plural, and drop the ending.
    4. Apply the rule to the last sound, and compare the result with the dictionary.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first cell holds the lesson's words as lists of sounds, `DUTCH` and `ENGLISH`, after the English Wiktionary as of November 22, 2026. Wiktionary writes *krabben* (crabs) as /ˈkrɑ.bə(n)/; the lists leave out the slashes, the stress mark ˈ, the dot between syllables and the final (n), which many speakers drop. A long vowel keeps its length mark ː, as in *zagen* (saws), and the vowel of *druif* (grape), œy̯, is one sound and one value. Wiktionary writes the Dutch h as ɦ, a voiced h, where the e-ANS writes h. `PAIRS` holds ten singulars with their plurals, and `VOICELESS` the five voiced obstruents with their voiceless partners. The second cell defines `stem`, which drops the last sound of a plural, its ending ə, and looks at a few values.
    """)
    return


@app.cell
def words_cell():
    DUTCH = {
        "hond": ["ɦ", "ɔ", "n", "t"], "honden": ["ɦ", "ɔ", "n", "d", "ə"],
        "hand": ["ɦ", "ɑ", "n", "t"], "handen": ["ɦ", "ɑ", "n", "d", "ə"],
        "hoed": ["ɦ", "u", "t"], "hoeden": ["ɦ", "u", "d", "ə"],
        "krab": ["k", "r", "ɑ", "p"], "krabben": ["k", "r", "ɑ", "b", "ə"],
        "rib": ["r", "ɪ", "p"], "ribben": ["r", "ɪ", "b", "ə"],
        "druif": ["d", "r", "œy̯", "f"], "druiven": ["d", "r", "œy̯", "v", "ə"],
        "zaag": ["z", "aː", "x"], "zagen": ["z", "aː", "ɣ", "ə"],
        "kat": ["k", "ɑ", "t"], "katten": ["k", "ɑ", "t", "ə"],
        "boek": ["b", "u", "k"], "boeken": ["b", "u", "k", "ə"],
        "kant": ["k", "ɑ", "n", "t"], "kanten": ["k", "ɑ", "n", "t", "ə"],
        "bed": ["b", "ɛ", "t"], "pet": ["p", "ɛ", "t"],
    }
    ENGLISH = {
        "pat": ["p", "æ", "t"], "bat": ["b", "æ", "t"], "pin": ["p", "ɪ", "n"], "bin": ["b", "ɪ", "n"],
        "spin": ["s", "p", "ɪ", "n"], "ship": ["ʃ", "ɪ", "p"], "sheep": ["ʃ", "iː", "p"],
        "think": ["θ", "ɪ", "ŋ", "k"], "sink": ["s", "ɪ", "ŋ", "k"],
    }
    PAIRS = [("hond", "honden"), ("hand", "handen"), ("hoed", "hoeden"), ("krab", "krabben"), ("rib", "ribben"),
             ("druif", "druiven"), ("zaag", "zagen"), ("kat", "katten"), ("boek", "boeken"), ("kant", "kanten")]
    VOICELESS = {"b": "p", "d": "t", "v": "f", "z": "s", "ɣ": "x"}
    return DUTCH, ENGLISH, PAIRS, VOICELESS


@app.cell
def example_look(DUTCH, VOICELESS):
    def stem(plural):
        return plural[:len(plural) - 1]

    print(DUTCH["honden"], stem(DUTCH["honden"]))
    print(DUTCH["hond"], len(DUTCH["hond"]))
    print(DUTCH["kat"][len(DUTCH["kat"]) - 1])
    print(VOICELESS["d"], "d" in VOICELESS, "t" in VOICELESS)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *Honden* without its ending gives the sounds ɦ ɔ n d: the d you hear in the plural. The singular *hond* has a t in that place. `len` counts values, so here it counts sounds, not letters. `DUTCH["kat"][len(DUTCH["kat"]) - 1]` is the last sound of *kat*, a t. `VOICELESS["d"]` gives `t`, and `in` asks whether a sound is a key of the dictionary: *t* is not, since it is voiceless already.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_minimal():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. minimal_pair: one sound apart

        Write `minimal_pair(a, b)`: `a` and `b` are two words as lists of sounds. Give back `True` when they form a minimal pair: the same number of sounds, and another sound in one place only. Otherwise give back `False`. For `DUTCH["hond"]` and `DUTCH["hand"]` that is `True`: only the vowel differs. Replace the `...`. The check calls `minimal_pair(DUTCH["hond"], DUTCH["hand"])`, then cases of its own: words of different lengths, words that differ in two places, and a word with itself.
        """),
        L.task("Tell whether two words, as lists of sounds, form a minimal pair."),
    ])
    return


@app.cell(hide_code=True)
def exercise_minimal():
    box_minimal = L.code_box("minimal_pair", lines=5)
    box_minimal
    return (box_minimal,)


@app.cell(hide_code=True)
def check_minimal(box_minimal):
    result_minimal = L.check("minimal_pair", box_minimal.value)
    result_minimal
    return (result_minimal,)


@app.cell(hide_code=True)
def hints_minimal():
    L.hints(
        "minimal_pair",
        [
            "Two words of different lengths are no minimal pair: give back `False` for them first, with an `if`. Then make a list `diff` with a list comprehension over the places `i` in `range(len(a))`, and look at its length.",
            "Which places belong in `diff`? And how long is `diff` for a minimal pair?",
            "`    diff = [i for i in range(len(a)) if ___]`",
        ],
        why="`if len(a) != len(b):` with `return False` under it, then `diff = [i for i in range(len(a)) if a[i] != b[i]]` and `return len(diff) == 1`. `minimal_pair(DUTCH[\"hond\"], DUTCH[\"hand\"])` gives `True`.",
    )
    return


@app.cell(hide_code=True)
def text_after_minimal():
    mo.md(r"""
    The picture under the check gives your answers for ten pairs. Six are minimal pairs, such as *hond* and *hand*, or English *ship* and *sheep*. *Kat* and *kant*, *hond* and *honden*, and *pin* and *spin* differ in length; *kat* and *pet* differ in two places. *Bed* and *pet* differ only in their first sound: *bed* ends in a t-sound, and exercise 2 shows why.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_devoice():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. devoice: the rule in code

        Write `devoice(sounds, voiceless)`: give back a list with the same sounds as `sounds`, but with the last one made voiceless when it is a key of the dictionary `voiceless`. A last sound that is no key stays as it is. For `stem(DUTCH["honden"])`, the sounds of *honden* without the ending, that is `['ɦ', 'ɔ', 'n', 't']`: the sounds of *hond*. Replace the `...`. The check calls `devoice(stem(DUTCH["honden"]), VOICELESS)`, then cases of its own: a voiced sound at the start, a last sound that is voiceless already, a word of one sound, and another dictionary.
        """),
        L.task("Make the last sound of a word voiceless, as the Dutch rule does."),
    ])
    return


@app.cell(hide_code=True)
def exercise_devoice():
    box_devoice = L.code_box("devoice", lines=7)
    box_devoice
    return (box_devoice,)


@app.cell(hide_code=True)
def check_devoice(box_devoice):
    result_devoice = L.check("devoice", box_devoice.value)
    result_devoice
    return (result_devoice,)


@app.cell(hide_code=True)
def hints_devoice():
    L.hints(
        "devoice",
        [
            "Make `out`, all sounds but the last, with a slice, and `last`, the last sound, as in the worked example. If `last` is a key of `voiceless`, give `last` a new value. Then add `last` to `out`, and give back `out`.",
            "Which sound should take the place of `last` when it is a key?",
            "`        last = ___`",
        ],
        why="`out = sounds[:len(sounds) - 1]` and `last = sounds[len(sounds) - 1]`; `if last in voiceless:` with `last = voiceless[last]` under it; then `out.append(last)` and `return out`. `devoice(stem(DUTCH[\"honden\"]), VOICELESS)` gives `['ɦ', 'ɔ', 'n', 't']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_devoice():
    mo.md(r"""
    The picture under the check takes each plural of `PAIRS`, drops its ending, and runs your `devoice` on what is left. For the seven words that end in d, b, v or ɣ before the ending, your rule gives the singular of the dictionary; the other three end in a voiceless sound already. Your function only looks at the end of the word. The rule itself works at the end of every syllable, as in *handrem*, and a voiced sound after it can overrule it, as in *huisdeur*.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: letters and sounds

    Choose a word from the lesson's two lists, and compare its letters with its sounds, and with the form as Wiktionary writes it.
    """)
    return


@app.cell(hide_code=True)
def demo_controls(DUTCH, ENGLISH):
    demo_word = mo.ui.dropdown(options=list(DUTCH) + list(ENGLISH), value="krabben", label="A word")
    demo_word
    return (demo_word,)


@app.cell(hide_code=True)
def demo_look(demo_word):
    mo.Html(C.demo_picture(demo_word.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Letters and sounds do not line up one to one. *Ship* has four letters and three sounds, since *sh* is one sound. *Krabben* writes one b-sound with two letters, and *druiven* writes one vowel as *ui*. *Hond* has as many letters as sounds, and still its last letter is not its last sound.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fields():
    mo.md(r"""
    ### Exercise 3. Sound, system or spelling?

    Each bold sentence below is about how a sound is made, about the sounds a language uses to keep words apart, or about how a language writes its sounds. Put each sentence under its group.
    """)
    return


@app.cell(hide_code=True)
def exercise_fields():
    sort_fields = L.sort("fields")
    sort_fields
    return (sort_fields,)


@app.cell(hide_code=True)
def check_fields(sort_fields):
    result_fields = L.check("fields", sort_fields)
    result_fields
    return (result_fields,)


@app.cell(hide_code=True)
def text_exercise_allophone():
    mo.md(r"""
    ### Exercise 4. One phoneme or two?

    In English, *pin* starts with [pʰ]: a p with a small puff of air after it. In *spin*, the p comes without the puff: [p].
    """)
    return


@app.cell(hide_code=True)
def choice_allophone_cell():
    choice_allophone = L.choice(
        "allophone",
        question="Why do the two count as one phoneme /p/ in English?",
        options={
            "letter": "Both are written with the letter *p*",
            "right": "Swapping them changes no English word",
            "puff": "The puff of air is too short to measure",
        },
        correct="right",
        why={
            "right": "In English, the two never tell words apart: said with the one or the other, a word stays the same word. They are allophones of /p/, and their place decides which one you hear, as with the two i's of *bier* and *biet*. The English Wikipedia gives this example, and the English Wiktionary writes *pin* with that detail as [ˈpʰɪn].",
            "letter": "Letters are no guide here: one letter can stand for two phonemes, as the *c* of *cell* and *cougar* shows. Look at what happens to a word when you swap the two sounds.",
            "puff": "The puff can be heard and measured; the IPA writes it with a small ʰ. Whether two sounds are one phoneme depends on what the language does with them. Look at what happens to a word when you swap them.",
        },
    )
    choice_allophone
    return (choice_allophone,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Words and their parts](/course/open/nlp-morphemes?lang=en) cut words into morphemes, along their letters. This lesson looked at the sounds under the letters: phonetics studies the sounds themselves, and phonology what a language does with them. A phoneme is a sound that can change a word, as a minimal pair shows; the IPA writes each sound with one symbol, as far as possible; and a rule such as final devoicing changes a sound by its place, so that the morpheme *hond* has two sound forms. You tested minimal pairs, wrote the rule in code, compared letters with sounds, and told phonetics, phonology and spelling apart. Speech recognition and text-to-speech start from sounds like these: the course's domain on speech and audio comes back to them.

    **An everyday example.** *Hart* (heart) and *hard* sound the same in Dutch: say them out loud. Which one is meant, you hear from the sentence, and read from the spelling.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_allophone, result_devoice, result_fields, result_minimal):
    L.wrap_up(result_minimal, result_devoice, result_fields, choice_allophone)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
