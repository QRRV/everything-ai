# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-keywords. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-keywords", "nl")
    C = L.checks   # de woorden van acht boeken, de 251 hoofdstukken van nlp-tf-idf, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_idf():
    quick_idf = L.choice(
        "quick-idf",
        question=r"In *TF-IDF: woorden wegen* krijgt een woord dat in $\text{df}$ van de $N$ documenten voorkomt de IDF $\log_{10}(N / \text{df})$. Met de 251 hoofdstukken als documenten: wat is de IDF van een woord dat in alle hoofdstukken voorkomt, zoals *the*?",
        options={
            "one": "1",
            "n": "251, het aantal hoofdstukken",
            "right": "0",
            "max": "Ongeveer 2,4: de hoogste IDF die er is, voor het vaakste woord",
        },
        correct="right",
        why={
            "right": r"$\log_{10}(251 / 251) = \log_{10} 1 = 0$: een woord in elk hoofdstuk zegt niets over één hoofdstuk in het bijzonder. Deze les neemt de woorden van een hoofdstuk met de hoogste TF-IDF als keywords, en kijkt daarna naar paren woorden.",
            "one": r"De IDF kijkt naar het deel van de hoofdstukken waarin het woord staat. Wat is 251 / 251, en wat is de logaritme daarvan? Zie [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl).",
            "n": r"251 / 251 is 1, en de IDF neemt daar de logaritme van. Wat is $\log_{10} 1$? Zie [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl).",
            "max": r"Ongeveer 2,4 is $\log_{10} 251$, de IDF van een woord dat in slechts één hoofdstuk voorkomt. Welk woord krijgt de laagste IDF? Zie [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl).",
        },
    )
    quick_idf
    return (quick_idf,)


@app.cell(hide_code=True)
def quick_check_weight():
    quick_weight = L.choice(
        "quick-weight",
        question="Een woord komt 4 keer voor in een document, en in 10 van de 1.000 documenten van een verzameling. Wat is de TF-IDF-waarde van het woord in dat document, met de IDF van *TF-IDF: woorden wegen*?",
        options={
            "times": "400: 4 keer 1.000 / 10",
            "right": "8",
            "idf": "2",
            "share": "0,04: het aantal 4 keer het deel van de documenten met het woord, 10 / 1.000",
        },
        correct="right",
        why={
            "right": r"De IDF is $\log_{10}(1000 / 10) = \log_{10} 100 = 2$, en de waarde is het aantal keer de IDF: 4 × 2 = 8. De keywords van een hoofdstuk zijn in deze les de woorden met de hoogste waarde.",
            "times": "1.000 / 10 is 100. Wat doet de IDF met dat getal 100, voordat het met het aantal vermenigvuldigd wordt? Zie [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl).",
            "idf": "2 is de IDF van het woord. Waarmee vermenigvuldigt TF-IDF de IDF? Zie [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl).",
            "share": r"TF-IDF vermenigvuldigt het aantal met $\log_{10}(N / \text{df})$. Welke kant op gaat dat quotiënt, en wat doet de logaritme ermee? Zie [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl).",
        },
    )
    quick_weight
    return (quick_weight,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Waar gaat een tekst over? De woorden die een tekst het vaakst gebruikt, zeggen weinig: *the*, *and* en *of* staan bovenaan in bijna elke Engelse tekst. In *TF-IDF: woorden wegen* woog je de woorden van een hoofdstuk tegen de andere hoofdstukken, zodat een woord in elk hoofdstuk niets kreeg. De woorden met de hoogste waarde zeggen waar het hoofdstuk over gaat: de keywords. Maar veel dingen waar een tekst over gaat, hebben een naam van twee woorden, zoals *Mock Turtle*, *Baker Street* of *natural philosophy*, en in een vaste uitdrukking zoals *of course* horen de twee woorden ook bij elkaar. Los zeggen de woorden minder dan het paar.

    In 1990 gebruikten Kenneth Church en Patrick Hanks een maat uit de informatietheorie om zulke paren te vinden door te tellen, in een paper over lexicografie, het maken van woordenboeken. Welke woorden bij elkaar horen, is een deel van wat een woordenboek over een woord moet zeggen. Deze les telt de paren naburige woorden in acht boeken, vijf in het Engels en drie in het Nederlands, en vindt de termen van elk boek.

    **Een voorbeeld uit het dagelijks leven.** Het toetsenbord van je telefoon stelt het volgende woord voor terwijl je typt: na *fijne* biedt het *feestdagen* aan. Het toetsenbord heeft geteld welke woorden vaak op welke volgen, en *fijne feestdagen* is een paar dat veel vaker voorkomt dan de twee woorden bij toeval zouden doen.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_chance = L.guess(
        "guess",
        question="*Alice* heeft 26.776 woorden, en 133 keer volgt *the* op *of*. Als dezelfde woorden in een willekeurige volgorde stonden, hoe vaak zou *the* dan op *of* volgen?",
        options={
            "same": "Ongeveer 130, zoals nu: een willekeurige volgorde houdt elk woord",
            "right": "Ongeveer 30",
            "few": "Ongeveer 3",
            "none": "Ongeveer 0,1: in een willekeurige volgorde komt een paar van twee bepaalde woorden bijna nooit terug",
        },
        reveal="**Ongeveer 32 keer.** *of* komt 515 keer voor en *the* 1.651 keer. In een willekeurige volgorde is het woord na een *of* met een kans van 1.651 op 26.776 een *the*, dus de 515 keer *of* geven ongeveer 515 × 1.651 / 26.776 = 31,8 paren *of the*. De echte 133 is ongeveer vier keer zoveel: *of the* komt vaak voor, vooral omdat *of* en *the* zelf vaak voorkomen. *mock turtle* zou bij toeval 0,12 keer voorkomen, en komt 54 keer voor: ongeveer 450 keer zoveel.",
    )
    guess_chance
    return (guess_chance,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een **keyword** van een document is een woord dat zegt waar het document over gaat. De eenvoudigste vorm van **keyword extraction** ken je al uit *TF-IDF: woorden wegen*: neem de woorden van het document met de hoogste TF-IDF-waarde, woorden die vaak in dit document voorkomen en in weinig andere. Voor het hoofdstuk *A Mad Tea-Party* van *Alice* staan dan *hatter*, *alice*, *dormouse*, *hare* en *march* bovenaan.

    *march* en *hare* zijn echter geen twee losse keywords: samen vormen de twee woorden de naam van de March Hare. Een **collocatie** is een paar woorden dat vaker samen voorkomt dan bij toeval, zoals *mock turtle*, *of course* of *sterke thee*. Sommige collocaties zijn namen, andere vaste uitdrukkingen, en veel zijn **termen**: de woorden waarmee een vakgebied de eigen dingen benoemt, zoals *natural philosophy* in *Frankenstein* of *quarter deck* in *Moby-Dick*.

    Hoe vaak is *vaker dan bij toeval*? Tel de woorden van een tekst: $N$ woorden in totaal, waarvan $c(x)$ keer het woord $x$ en $c(y)$ keer het woord $y$, en $c(x, y)$ paren waarin direct na $x$ het woord $y$ komt. De kans dat een woord $x$ is, is $P(x) = c(x) / N$, en de kans dat een paar $x$ gevolgd door $y$ is, is $P(x, y) = c(x, y) / N$. Twee woorden zijn **onafhankelijk** als het ene niets zegt over het andere: dan is de kans op het paar de kans op het ene keer de kans op het andere, $P(x) \cdot P(y)$. Voor onafhankelijke woorden, zoals in een willekeurige volgorde, zou het paar ongeveer

    $$\text{aantal bij toeval} = N \cdot P(x) \cdot P(y) = \frac{c(x) \cdot c(y)}{N}$$

    keer voorkomen. De **pointwise mutual information** van het paar, kort PMI, vergelijkt het echte aantal met het aantal bij toeval:

    $$\text{PMI}(x, y) = \log_2 \frac{P(x, y)}{P(x) \cdot P(y)} = \log_2 \frac{c(x, y)}{c(x) \cdot c(y) / N}$$

    Church en Hanks beschrijven mutual information informeel als een vergelijking van twee kansen: de kans dat je $x$ en $y$ samen ziet, de gezamenlijke kans, en de kansen dat je $x$ en $y$ los van elkaar ziet, het toeval. Als $x$ en $y$ echt bij elkaar horen, is de gezamenlijke kans $P(x, y)$ volgens Church en Hanks veel groter dan het toeval $P(x)\,P(y)$, en dan is $I(x, y)$ veel groter dan 0. Church en Hanks schrijven $I(x, y)$ voor wat nu PMI heet, en tellen zowel de woorden als de paren uit $N$, de grootte van het corpus, net als deze les.

    De logaritme met grondtal 2 maakt van *twee keer zo vaak als bij toeval* een 1, van vier keer een 2, en van 64 keer een 6. Een PMI van 0 betekent precies zo vaak als bij toeval, en een paar dat minder vaak voorkomt dan bij toeval, krijgt een PMI onder 0.

    De afbeelding laat allebei zien: de keywords van *A Mad Tea-Party* met TF-IDF, en zes paren uit heel *Alice*, elk met het aantal, het aantal bij toeval en de PMI.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    *of the* komt ongeveer vier keer zo vaak voor als bij toeval, een PMI van 2,07; *mock turtle* en *march hare* honderden keren zo vaak. Het laatste paar, *abide figures*, uit een zin van de Duchess, *I never could abide figures!*, komt slechts één keer voor, en krijgt toch de hoogste PMI van de zes. Oefening 3 en 4 komen daarop terug.
    """)
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### Nieuwe woorden en tekens

    | Woord | Betekenis |
    |---|---|
    | **keyword** | een woord dat zegt waar een document over gaat; hier: een van de woorden met de hoogste TF-IDF |
    | **keyword extraction** | de keywords van een document met de computer vinden |
    | **collocatie** | een paar woorden dat vaker samen voorkomt dan bij toeval, zoals *mock turtle* of *of course* |
    | **term** | een woord of een vaste groep woorden die in een vakgebied één ding noemt, zoals *natural philosophy* |
    | **onafhankelijk** | twee woorden zijn onafhankelijk als het ene niets zegt over het andere: $P(x, y) = P(x) \cdot P(y)$ |
    | **aantal bij toeval** | hoe vaak een paar zou voorkomen als de woorden onafhankelijk waren: $c(x) \cdot c(y) / N$ |
    | **pointwise mutual information** (PMI) | $\log_2$ van hoeveel keer zo vaak als bij toeval een paar voorkomt: 0 voor even vaak, 1 voor twee keer zo vaak, onder 0 voor minder vaak |
    | `{("mock", "turtle"): 54}` | een tuple kan een key van een dictionary zijn, een lijst niet |
    | `x, y = pair` | geeft elke waarde van een tuple van twee een eigen naam |
    | `sorted(scores, reverse=True)` | sorteert een lijst van groot naar klein; tuples worden vergeleken op het eerste element, en bij gelijke stand op het volgende |
    | `C.book_words(key)` | de woorden van een boek in kleine letters, op volgorde, zoals `C.book_words("alice")`; de keys van de acht boeken staan in `C.BOOKS` |
    | `C.word_counts(key)`, `C.book_pairs(key)` | hoe vaak elk woord, en elk paar naburige woorden, in een boek voorkomt |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Tel de woorden van de tekst, en de paren naburige woorden.
    2. Reken voor elk paar het aantal bij toeval uit: $c(x) \cdot c(y) / N$.
    3. De PMI is $\log_2$ van het echte aantal gedeeld door het aantal bij toeval.
    4. Houd de paren die minstens een minimum aantal keer voorkomen, en zet de hoogste PMI eerst: dat zijn de termen van de tekst.
    5. Neem voor keywords van één woord de woorden met de hoogste TF-IDF.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Church en Hanks geven een voorbeeld uit 44.344.077 woorden nieuws van persbureau Associated Press uit 1988: *set* komt er 13.046 keer in voor, *off* 20.693 keer, en *set off* 463 keer. De code rekent het aantal bij toeval van *set off* uit, hoeveel keer zo vaak het paar voorkomt, en de PMI. Daarna sorteert de code drie rijen met PMI en paar uit *Alice* vanaf de hoogste PMI.
    """)
    return


@app.cell
def example_cell():
    import numpy as np

    _n = 44_344_077
    _chance = 13_046 * 20_693 / _n
    print(round(_chance, 1))
    print(round(463 / _chance, 1))
    print(round(np.log2(463 / _chance), 2))

    _scores = [(2.07, ("of", "the")), (8.80, ("mock", "turtle")), (4.12, ("said", "alice"))]
    print(sorted(_scores, reverse=True))
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Bij toeval zou *set off* ongeveer 6 keer voorkomen; het paar komt 76 keer zo vaak voor, en $\log_2 76{,}1 = 6{,}25$. Church en Hanks noemen ongeveer 6,2, en leggen uit wat dat betekent: de gezamenlijke kans is meer dan $2^6 = 64$ keer zo groot als het toeval. `sorted` met `reverse=True` zet de rijen vanaf de hoogste PMI: de functie vergelijkt het eerste element van elk tuple, de PMI, en zou alleen bij gelijke stand naar de paren kijken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pairs():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. pair_counts: de paren van een tekst

        Schrijf `pair_counts(words)`, waarbij `words` een lijst woorden is. Geef een dictionary terug van elk paar naburige woorden, een tuple van een woord en het woord erna, naar hoe vaak dat paar in `words` voorkomt. De loop gaat langs elke plaats `i`, vanaf 0 geteld, met een woord erna. Vervang de twee keer `...`: het paar op plaats `i`, en de regels die 1 optellen bij het aantal in `pairs`, of 1 neerzetten voor een paar dat nog niet in `pairs` staat.

        De check roept `pair_counts(song)` aan, met de woorden `["beau", "ootiful", "soo", "oop", "beau", "ootiful", "soo", "oop"]` uit het lied van de Mock Turtle, twee keer *Beau—ootiful Soo—oop!* (in de les: `C.SONG`). Dat geeft `{("beau", "ootiful"): 2, ("ootiful", "soo"): 2, ("soo", "oop"): 2, ("oop", "beau"): 1}`. Daarna eigen gevallen: een lege lijst; `["alice"]`; `["ha", "ha", "ha"]`; en alle woorden van *Alice* en van *Max Havelaar* (in de les: `C.book_words("alice")` en `C.book_words("havelaar")`).
        """),
        L.task("Tel de paren naburige woorden in een tekst."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pairs():
    box_pairs = L.code_box("pair_counts", lines=10)
    box_pairs
    return (box_pairs,)


@app.cell(hide_code=True)
def check_pairs(box_pairs):
    result_pairs = L.check("pair_counts", box_pairs.value)
    result_pairs
    return (result_pairs,)


@app.cell(hide_code=True)
def hints_pairs():
    L.hints(
        "pair_counts",
        [
            "In de loop is `pair` een tuple van twee woorden uit `words`: het woord op plaats `i` en het woord erna. Daarna telt `if pair in pairs:` 1 op bij het aantal, en zet `else:` er 1 neer.",
            'Ronde haakjes met een komma ertussen maken een tuple, zoals `("mock", "turtle")`. Welke plaats in `words` heeft het woord na het woord op plaats `i`, en hoe tel je 1 op bij een waarde in een dictionary?',
            "`        pair = (words[i], ___)`",
        ],
        why='`pair = (words[i], words[i + 1])`, daarna `if pair in pairs:` met `pairs[pair] = pairs[pair] + 1` eronder, en `else:` met `pairs[pair] = 1`. Het voorbeeld geeft `{("beau", "ootiful"): 2, ("ootiful", "soo"): 2, ("soo", "oop"): 2, ("oop", "beau"): 1}`.',
    )
    return


@app.cell(hide_code=True)
def text_after_pairs():
    mo.md(r"""
    *Alice* heeft 26.776 woorden en 14.752 verschillende paren, en 11.234 daarvan komen slechts één keer voor. De vaakste paren, *said the*, *of the* en *said alice*, bestaan uit de vaakste woorden.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pmi():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. pmi: tegen toeval

        Schrijf `pmi(pair, pairs, counts, n)`, waarbij `pair` een tuple van twee woorden is, `pairs` de aantallen van de paren van een tekst, als uitkomst van je `pair_counts`, `counts` de aantallen van de woorden, en `n` het aantal woorden. Geef de PMI van het paar terug: $\log_2$ van het aantal gedeeld door het aantal bij toeval, met `np.log2`. De eerste regel geeft de twee woorden van het paar de namen `x` en `y`. Vervang de twee keer `...`.

        De check roept `pmi(("mock", "turtle"), alice_pairs, alice_counts, n_alice)` aan, met de paren en de woorden van *Alice* en het aantal woorden, 26.776 (in de les: `C.book_pairs("alice")`, `C.word_counts("alice")` en `len(C.book_words("alice"))`). Dat geeft ongeveer 8,80. Daarna eigen gevallen: *of the*, *said alice*, *abide figures* en *twinkle twinkle* in *Alice*; het paar `("a", "b")` van de woorden *a b a b*, met de paren `{("a", "b"): 2, ("b", "a"): 1}`, de aantallen `{"a": 2, "b": 2}` en 4 woorden, wat 1,0 geeft; en *and the* in *Moby-Dick*.
        """),
        L.task("Reken uit hoeveel keer zo vaak als bij toeval een paar voorkomt, als PMI."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pmi():
    box_pmi = L.code_box("pmi", lines=8)
    box_pmi
    return (box_pmi,)


@app.cell(hide_code=True)
def check_pmi(box_pmi):
    result_pmi = L.check("pmi", box_pmi.value)
    result_pmi
    return (result_pmi,)


@app.cell(hide_code=True)
def hints_pmi():
    L.hints(
        "pmi",
        [
            "`chance` is het aantal van `x` in `counts` keer het aantal van `y`, gedeeld door `n`. Daarna geeft `return` de `np.log2` van het aantal van het paar in `pairs`, gedeeld door `chance`.",
            "Voor *mock turtle*: 57 keer *mock*, 57 keer *turtle* en 26.776 woorden geven een aantal bij toeval van 0,12, en het paar komt 54 keer voor. Hoe schrijf je het aantal van het woord `x` uit de dictionary `counts`?",
            "`    chance = ___ / n`",
        ],
        why="`chance = counts[x] * counts[y] / n` en `return np.log2(pairs[pair] / chance)`. Het voorbeeld geeft 8,80: *mock turtle* komt 445 keer zo vaak voor als bij toeval.",
    )
    return


@app.cell(hide_code=True)
def text_after_pmi():
    mo.md(r"""
    Church en Hanks geven een heel ruwe vuistregel uit eigen ervaring: paren met een $I(x, y)$ boven 3 zijn vaak interessant, en paren met een lagere waarde meestal niet. In je afbeelding zitten *of the* (2,07) en *and the* (0,61) onder 3, en *the queen* (3,95) en *said alice* (4,12) erboven. In *Moby-Dick* komt *and the* 372 keer voor, waar toeval 432 zou geven: een PMI van −0,22, minder vaak dan bij toeval.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_terms():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. terms: de termen van een boek

        Schrijf `terms(pairs, counts, n, least, k)`, met `pairs`, `counts` en `n` zoals in oefening 2. Geef de `k` paren met de hoogste PMI terug van de paren die minstens `least` keer voorkomen, als een lijst van tuples `(PMI, paar)`, vanaf de hoogste PMI; bij gelijke PMI zet `sorted` met `reverse=True` het paar dat later in het alfabet komt eerst. In het vak is `pmi` de functie van de course, die net zo werkt als jouw `pmi` uit oefening 2. Vervang de twee keer `...`: in de loop de regels die de rij van een paar dat vaak genoeg voorkomt aan `rows` toevoegen, en na de loop de eerste `k` rijen vanaf de hoogste PMI.

        De check roept `terms(alice_pairs, alice_counts, n_alice, 10, 5)` aan: de termen van *Alice* van de paren die minstens 10 keer voorkomen. Dat geeft de rijen van *march hare* (9,62), *mock turtle* (8,80), *your majesty* (8,75), *white rabbit* (8,65) en *minute or* (7,63), in deze volgorde, elk een tuple zoals `(9.62…, ("march", "hare"))`. Daarna eigen gevallen: in *Alice* minstens 1 met `k` 3, minstens 20 met `k` 5, minstens 1.000 met `k` 5 (geen enkel paar komt zo vaak voor) en minstens 10 met `k` 0; de paren van *a b a b* uit oefening 2 met minstens 1 en `k` 2; en minstens 20 met `k` 5 in *Moby-Dick* en in *Max Havelaar*.
        """),
        L.task("Vind de termen van een boek: de paren met de hoogste PMI die vaak genoeg voorkomen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_terms():
    box_terms = L.code_box("terms", lines=8)
    box_terms
    return (box_terms,)


@app.cell(hide_code=True)
def check_terms(box_terms):
    result_terms = L.check("terms", box_terms.value)
    result_terms
    return (result_terms,)


@app.cell(hide_code=True)
def hints_terms():
    L.hints(
        "terms",
        [
            "In de loop houdt een `if` de paren waarvan het aantal in `pairs` minstens `least` is, en `rows.append` voegt een tuple toe van de PMI van het paar, uit `pmi`, en het paar. Na de loop geeft `return` de eerste `k` rijen van `rows`, gesorteerd met `reverse=True`.",
            "Een tuple `(PMI, paar)` wordt eerst op de PMI gesorteerd. Welke aanroep zet de rijen van groot naar klein, en hoe neem je de eerste `k` elementen van een lijst?",
            "`    return ___[:k]`",
        ],
        why="In de loop: `if pairs[pair] >= least:` met `rows.append((pmi(pair, pairs, counts, n), pair))` eronder; na de loop `return sorted(rows, reverse=True)[:k]`. Het voorbeeld geeft *march hare*, *mock turtle*, *your majesty*, *white rabbit* en *minute or*.",
    )
    return


@app.cell(hide_code=True)
def text_after_terms():
    mo.md(r"""
    Je afbeelding laat zien hoeveel het minimum uitmaakt. Met minstens 1 staan bovenaan alleen paren die één keer voorkomen, zoals *wooden spades* en *tittered audibly*. Met minstens 3 gaan *yer honour*, *mary ann* en *beau ootiful* voorop. Met minstens 10 en 20 komen de namen en vaste uitdrukkingen van het boek boven: *march hare*, *mock turtle*, *white rabbit*, *did not*, *went on*. *minute or* komt uit *in a minute or two*: met alleen buren verschijnt een vaste uitdrukking van drie woorden of meer als losse paren. Ook Church en Hanks gebruikten een minimum: omdat de maat bij kleine aantallen onbetrouwbaar wordt, bespreekt de paper geen paren die 5 keer of minder voorkomen, met een grens die de paper zelf eenvoudig maar willekeurig noemt. Church en Hanks telden ook $x$ gevolgd door $y$ binnen vijf woorden, niet alleen als buren, en schrijven dat een kleiner venster vaste uitdrukkingen vindt, zoals *bread and butter*.
    """)
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het: de termen van een boek

    Kies een boek, een minimum aantal en een volgorde. De tabel laat de twaalf paren met de hoogste PMI zien, of de vaakste paren, van de paren die minstens zo vaak voorkomen, elk met het aantal, het aantal bij toeval en de PMI. In welk boek vullen namen de top, en in welk vaste uitdrukkingen? Wat hebben de vaakste paren gemeen, en hoe hoog is de PMI ervan? Welk minimum geeft voor elk boek de beste lijst? De drie Nederlandse boeken komen uit de negentiende eeuw en zijn nog in de oude spelling geschreven, zoals *mensch* voor *mens*.
    """)
    return


@app.cell(hide_code=True)
def try_controls():
    try_book = mo.ui.dropdown(options=C.book_options(L.lang), value=C.book_label("alice", L.lang), label="Boek")
    try_least = mo.ui.slider(1, 30, value=1, label="Minstens", show_value=True)
    try_order = mo.ui.radio(options={"hoogste PMI": "pmi", "vaakst": "count"}, value="hoogste PMI", label="Volgorde", inline=True)
    mo.hstack([try_book, try_least, try_order], justify="start", gap=2, wrap=True)
    return try_book, try_least, try_order


@app.cell(hide_code=True)
def try_look(try_book, try_least, try_order):
    mo.Html(C.try_picture(try_book.value, try_least.value, try_order.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_once():
    mo.md(r"""
    ### Oefening 4. Paren die één keer voorkomen

    Zonder minimum staan bovenaan de PMI in elk boek paren die één keer voorkomen: in *Alice* komen de 71 paren met de hoogste PMI allemaal één keer voor, zoals *wooden spades*, en elk krijgt 14,71, veel meer dan *march hare* met 9,62.
    """)
    return


@app.cell(hide_code=True)
def choice_once_cell():
    why_once = L.choice(
        "why_once",
        question="Waarom krijgen deze paren die één keer voorkomen de hoogste PMI?",
        options={
            "rare": "PMI beloont zeldzame paren met opzet: Church en Hanks wilden de zeldzame paren, omdat een woordenboek de zeldzame paren het meest nodig heeft",
            "right": "De twee woorden komen elk één keer voor, dus het aantal bij toeval is 1 / N en het quotiënt N, het grootste dat er is",
            "round": "Afronding: met aantallen van 1 is de logaritme niet precies uit te rekenen",
            "log": "De logaritme van een aantal van 1 is 0, en PMI draait dat om tot de hoogste waarde",
        },
        correct="right",
        why={
            "right": r"Voor *abide figures* is $c(x) = c(y) = c(x, y) = 1$, dus het aantal bij toeval is $1 \cdot 1 / 26.776$, en het paar komt 26.776 keer zo vaak voor als bij toeval: $\log_2 26.776 = 14{,}71$, de hoogste PMI die een paar van *Alice* kan krijgen. Eén toeval is genoeg. Een minimum aantal, zoals in je `terms`, houdt zulke toevallen uit de top.",
            "rare": "Church en Hanks deden het omgekeerde: de paper laat paren met kleine aantallen weg. Wat doet een aantal van 1 met het quotiënt van PMI?",
            "round": "De getallen zijn exact: 1 · 26.776 / (1 · 1) is 26.776. Wat is het grootste quotiënt dat een paar kan krijgen?",
            "log": "PMI neemt de logaritme van het quotiënt van het echte aantal en het aantal bij toeval, niet van het aantal alleen. Wat is dat quotiënt voor een paar waarvan de woorden elk één keer voorkomen?",
        },
    )
    why_once
    return (why_once,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl) woog je de woorden van een document tegen de andere documenten; hier werden de woorden met de hoogste waarde de keywords. Daarna keek je naar paren: je `pair_counts` telde de paren naburige woorden, je `pmi` vergeleek het aantal van een paar met het aantal dat toeval zou geven, en je `terms` hield de paren die vaak genoeg voorkomen en zette de hoogste PMI eerst. De woorden van een tekst zijn niet onafhankelijk: na *mock* komt bijna altijd *turtle*. PMI meet hoe ver een paar van onafhankelijk af zit, en een minimum aantal houdt paren die één keer bij toeval voorkomen uit de top. De volgende les, *Gegevens uit tekst halen met een LLM*, laat een LLM de feiten uit een tekst halen.

    **Een voorbeeld uit het dagelijks leven.** Een winkel die wil voorstellen wat bij een product past, kan tellen welke producten vaker dan bij toeval in hetzelfde mandje belanden. Brood en melk liggen vaak samen in een mandje, maar vooral omdat brood en melk elk in zoveel mandjes liggen; pasta en geraspte kaas samen zeggen meer.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_pairs, result_pmi, result_terms, why_once):
    L.wrap_up(result_pairs, result_pmi, result_terms, why_once)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
