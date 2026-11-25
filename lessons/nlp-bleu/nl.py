# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-bleu. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-bleu", "nl")
    C = L.checks   # de vertalingen van nlp-classic-mt en van mensen, BLEU en chrF, de plaatjes en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl), en één over [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bigrams():
    quick_bigrams = L.choice(
        "quick-bigrams",
        question="In *Een bigram-model door te tellen* is een bigram twee tokens naast elkaar, en je telde de bigrams van een tekst teken voor teken. Hoeveel bigrams heeft *banana*, als je elk bigram telt zo vaak als het voorkomt?",
        options={
            "distinct": "3, één voor elk verschillend paar: *ba*, *an* en *na*",
            "right": "5",
            "letters": "6, één voor elke letter",
            "four": "4",
        },
        correct="right",
        why={
            "right": "*banana* heeft 6 letters, en op elk van de eerste 5 begint een bigram: *ba*, *an*, *na*, *an*, *na*; *an* en *na* komen twee keer voor. Deze les telt zulke n-grams, van woorden en van letters, en hoeveel daarvan een vertaling deelt met een vertaling door mensen.",
            "distinct": "3 telt elk verschillend paar één keer, en *an* en *na* komen elk twee keer voor. Op hoeveel plaatsen begint een bigram? Zie [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl).",
            "letters": "De laatste letter, *a*, begint geen bigram: er komt niets meer na. Op hoeveel plaatsen begint een bigram? Zie [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl).",
            "four": "*banana* heeft 6 letters, en elke letter behalve de laatste begint een bigram. Hoeveel is dat? Zie [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl).",
        },
    )
    quick_bigrams
    return (quick_bigrams,)


@app.cell(hide_code=True)
def quick_check_recall():
    quick_recall = L.choice(
        "quick-recall",
        question="In *Meer dan accuracy: precision en recall* is recall het deel van wat positief is dat een model vindt. Een spamfilter markeert 10 berichten als spam, en 8 daarvan zijn spam. In totaal zijn 20 berichten spam. Wat is de recall?",
        options={
            "precision": "0,8, want 8 van de 10 gemarkeerde berichten zijn spam",
            "half": "0,5",
            "right": "0,4",
            "two": "0,2",
        },
        correct="right",
        why={
            "right": "Recall is 8 / 20 = 0,4: van de 20 spamberichten vindt het filter er 8. De precision is 8 / 10 = 0,8. Deze les berekent allebei voor de letters van een vertaling: precision is het deel van de groepjes letters van de vertaling dat ook in de vertaling door mensen staat, en recall het deel van de groepjes letters van de vertaling door mensen dat de vertaling heeft.",
            "precision": "8 / 10 = 0,8 is de precision: van wat het filter markeert, het deel dat spam is. Welk deel van de 20 spamberichten vindt het filter? Zie [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
            "half": "10 / 20 vergelijkt hoeveel berichten het filter markeert met hoeveel er spam zijn, en 2 van de 10 gemarkeerde berichten zijn geen spam. Hoeveel van de 20 spamberichten vindt het filter? Zie [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
            "two": "2 van de 10 gemarkeerde berichten zijn geen spam: dat is het deel fouten in wat het filter markeert. Hoeveel van de 20 spamberichten vindt het filter? Zie [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
        },
    )
    quick_recall
    return (quick_recall,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In *Vertalen met regels* was een vertaling precies een vertaling door mensen, stonden de woorden in een andere volgorde, of waren het andere woorden. Van de 21.636 vertalingen woord voor woord hadden er 17.620 andere woorden: *i have hunger*, waar mensen *I'm hungry* schrijven, telt even zwaar als een tekst zonder één goed woord. Mensen kunnen elke vertaling beoordelen, maar dat is traag en kost veel. In 2002 stelden Kishore Papineni en drie collega's van IBM BLEU voor, een score die een computer in een oogwenk uitrekent. Het idee erachter: hoe dichter een machinevertaling bij een vertaling van een professionele vertaler ligt, hoe beter. Papineni en collega's maten hoe dichtbij een vertaling ligt door de groepjes woorden te tellen die de vertaling deelt met vertalingen door mensen, en vergeleken de score met het oordeel van mensen over dezelfde vertalingen. Met een correlatiecoëfficiënt van 0,99 volgt BLEU dat oordeel goed, schreven de onderzoekers. Bij een correlatiecoëfficiënt van 1 zouden BLEU en het oordeel van mensen precies gelijk op stijgen en dalen.

    Ook waarschuwden Papineni en collega's: BLEU hoeft alleen gemiddeld over een hele testverzameling met het oordeel van mensen overeen te komen, en de scores van losse zinnen wijken vaak af van wat mensen vinden. In deze les bouw je BLEU, en chrF, een score die groepjes letters telt, en scoor je de vertalingen van *Vertalen met regels* met allebei.

    **Een voorbeeld uit het dagelijks leven.** Een leraar die een dictee nakijkt, rekent niet de hele tekst goed of fout, maar telt de woorden die goed gespeld zijn: één verkeerde letter kost een beetje, niet alles.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_zero = L.guess(
        "guess",
        question="De les heeft 3.707 Nederlandse zinnen van Tatoeba met meer dan één Engelse vertaling door mensen. Neem bij elke zin de eerste vertaling, en scoor deze vertaling zin voor zin tegen de andere, met BLEU in de vorm van Papineni en collega's. BLEU telt de groepjes van 1 tot 4 woorden op een rij die een vertaling met de andere deelt, en geeft 0 als de vertaling geen enkel groepje van vier deelt. Bij hoeveel van de 3.707 geeft BLEU 0?",
        options={
            "almost": "Bijna geen",
            "tenth": "Ongeveer één op de tien",
            "right": "Meer dan de helft",
            "all": "Bijna allemaal",
        },
        reveal="Meer dan de helft: 2.200 van de 3.707, of 59,3%. 775 van de eerste vertalingen hebben minder dan vier woorden, dus helemaal geen groepje van vier. 193 daarvan zijn precies gelijk aan een andere vertaling: *Ik ben moe.* heeft *I'm tired.* en *I am tired.*, dezelfde woorden als je *I'm* uitschrijft, en BLEU geeft 0. De andere 1.425 hebben vier woorden of meer, maar delen geen groepje van vier met een andere vertaling: *Kan ik u helpen?* heeft *Can I help you?* en *May I help you?*, die *i help you* delen, drie woorden. Allebei goed Engels: mensen zeggen één ding op veel goede manieren.",
    )
    guess_zero
    return (guess_zero,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een vertaling door mensen waarmee je een vertaling vergelijkt, heet een **referentie**; een zin kan er meer dan één hebben. BLEU, kort voor *bilingual evaluation understudy*, telt de **n-grams van woorden** die een vertaling met de referenties deelt: groepjes van 1, 2, 3 en 4 woorden op een rij, zoals in *Een bigram-model door te tellen*. Vier stappen maken van de tellingen een score:

    1. **Afgetopte tellingen** (clipped counts). Tel voor elke n de n-grams van de vertaling die in een referentie staan, maar elk hoogstens zo vaak als één referentie het heeft. Papineni en collega's toppen de telling van elk woord van de vertaling af op het hoogste aantal in één referentie, en langere n-grams gaan net zo. De afgetopte telling gedeeld door het aantal n-grams van de vertaling is de **n-gram-precision** $p_n$.
    2. **Het meetkundig gemiddelde** van $p_1$ tot en met $p_4$: vermenigvuldig de vier en neem de vierdemachtswortel, of, hetzelfde, neem *e* tot de macht het gemiddelde van de logaritmen. Papineni en collega's merken op dat het gewone gemiddelde van de logaritmen neerkomt op het meetkundig gemiddelde van de precisions. Anders dan het gewone gemiddelde is het meetkundig gemiddelde 0 zodra één $p_n$ 0 is.
    3. **De brevity penalty** (BP). Precision alleen beloont een vertaling waarin woorden ontbreken: *the cat is on* heeft alleen woorden van *the cat is on the mat*. Met $c$ de lengte van de vertaling in woorden en $r$ de lengte van de referentie die qua lengte het dichtst bij de vertaling ligt: $\text{BP} = 1$ als $c > r$, en $\text{BP} = e^{1 - r/c}$ als $c \le r$, en dat is kleiner dan 1 als $c < r$.
    4. **BLEU** is BP keer het meetkundig gemiddelde, van 0 tot 1. Hulpmiddelen als sacreBLEU, een Python-package dat BLEU en chrF berekent, en deze les, schrijven BLEU keer 100: 22,9 voor 0,229.

    Bij veel zinnen telt BLEU eerst de tellingen en lengtes van alle zinnen op, en deelt pas daarna: BLEU op **corpusniveau**. Papineni en collega's berekenen de brevity penalty over het hele corpus, om op het niveau van een zin wat vrijheid te laten.

    BLEU vergelijkt hele woorden. Het Nederlands schrijft veel **samenstellingen** als één woord: *insectensoorten* is *insecten* en *soorten*, en voor BLEU is het een woord dat op geen van beide lijkt. In 2015 stelde Maja Popović **chrF** voor, een score die **n-grams van tekens** telt: groepjes van 1 tot 6 letters op een rij, zonder de spaties. Per lengte is precision het deel van de n-grams van tekens van de vertaling dat in de referentie staat, afgetopt zoals bij BLEU, en recall het deel van de n-grams van tekens van de referentie dat de vertaling heeft. chrF middelt allebei over de zes lengtes, tot $P$ en $R$, en combineert $P$ en $R$ in een **F-score**:

    $$\text{chrF}_\beta = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 \cdot P + R}$$

    Volgens Popović geeft β recall β keer zoveel belang als precision. Met β = 1 tellen precision en recall even zwaar. In 2016 schreef Popović dat CHRF2 de meest veelbelovende versie was: β = 2, de waarde die deze les en sacreBLEU gebruiken.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **referentie** | een vertaling door mensen waarmee je een vertaling vergelijkt; een zin kan er meer dan één hebben |
    | **n-gram van woorden**, **n-gram van tekens** | n woorden op een rij, of n tekens op een rij; deze les laat de spaties weg uit n-grams van tekens |
    | **afgetopte telling** (clipped count) | hoe vaak de n-grams van een vertaling in een referentie staan, elk hoogstens zo vaak als één referentie het heeft |
    | **n-gram-precision** ($p_n$) | de afgetopte telling van de n-grams van een vertaling, gedeeld door het aantal n-grams van de vertaling |
    | **meetkundig gemiddelde** | van k getallen: het product tot de macht 1/k; hetzelfde als *e* tot de macht het gemiddelde van de logaritmen |
    | **brevity penalty** (BP) | een factor van hoogstens 1 die BLEU verlaagt bij een vertaling die korter is dan de referentie: $e^{1 - r/c}$ als $c \le r$ |
    | **corpusniveau** | de tellingen van alle zinnen optellen en pas dan delen, in plaats van zin voor zin te scoren |
    | **BLEU** | de brevity penalty keer het meetkundig gemiddelde van $p_1$ tot en met $p_4$ |
    | **samenstelling** | een woord dat uit twee of meer woorden bestaat, zoals *insectensoorten* |
    | **F-score** ($F_\beta$) | één getal uit precision $P$ en recall $R$ dat recall β keer zo zwaar telt als precision |
    | **chrF** | de F-score met β = 2 van de n-grams van tekens van lengte 1 tot en met 6, met precision en recall gemiddeld over de lengtes |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om vertalingen met BLEU te scoren:

    1. Knip elke vertaling en elke referentie in woorden.
    2. Tel voor n = 1 tot en met 4 de n-grams van elke vertaling, en top elke telling af op de hoogste telling in één referentie.
    3. Tel over alle zinnen de afgetopte tellingen op, en ook de aantallen n-grams, de lengtes van de vertalingen en de lengtes van de dichtstbijzijnde referenties.
    4. Deel: $p_1$ tot en met $p_4$. Neem het meetkundig gemiddelde, en vermenigvuldig met de brevity penalty.

    Om vertalingen met chrF te scoren, laat je de spaties weg uit de woorden, tel je de n-grams van tekens van lengte 1 tot en met 6 in elke vertaling en de referentie, en top je af. Tel op over alle zinnen, bereken precision en recall per lengte, middel allebei over de lengtes, en neem de F-score met β = 2. Een zin met meer dan één referentie neemt de referentie met de hoogste chrF.

    Deze les scoort woorden in kleine letters zonder leestekens, met *I'll* uitgeschreven als *i will*, zoals in *Vertalen met regels*. Op deze woorden geeft sacreBLEU voor alle zinnen samen dezelfde scores als de course; voor een losse zin doet sacreBLEU iets extra's, zoals de tekst na oefening 2 vertelt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Neem het voorbeeld van Papineni en collega's: de vertaling *the the the the the the the.*, met als eerste referentie *The cat is on the mat.* en als tweede *There is a cat on the mat.* `C.words` knipt een tekst in woorden in kleine letters, en `C.ngram_counts` telt de n-grams van een lijst met woorden in een dictionary, elk n-gram geschreven als één tekst, met spaties tussen de woorden:
    """)
    return


@app.cell
def example_cell():
    _mine = C.ngram_counts(C.words("the the the the the the the."), 1)
    _first = C.ngram_counts(C.words("The cat is on the mat."), 1)
    _second = C.ngram_counts(C.words("There is a cat on the mat."), 1)
    print(_mine)
    print(_first["the"], _second["the"])
    print(C.ngram_counts(C.words("The cat is on the mat."), 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De vertaling heeft 7 unigrams, allemaal *the*, en beide referenties hebben *the*: gewoon geteld is de precision 7/7. Maar de eerste referentie heeft *the* twee keer en de tweede één keer, dus *the* telt hoogstens twee keer: Papineni en collega's komen zo op een aangepaste, dus afgetopte, unigram-precision van 2/7. De 6 bigrams zijn allemaal *the the*, en geen referentie heeft *the the*: $p_2 = 0/6$, dus BLEU is 0. Bij meer dan één referentie geeft de course je `clipped` één dictionary met de hoogste telling van elk n-gram in één referentie: hier *the* 2 keer, uit de eerste referentie, en *there* één keer, uit de tweede.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_clipped():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. clipped: tel wat de referentie heeft

        Schrijf `clipped`, dat `mine` krijgt, de tellingen van de n-grams van een vertaling, en `theirs`, de hoogste telling van elk n-gram in één referentie, allebei dictionaries zoals `C.ngram_counts` geeft. Geef terug hoeveel n-grams van de vertaling tellen: een n-gram van `mine` dat in `theirs` staat, telt zo vaak als `mine` het heeft, maar nooit vaker dan `theirs` het heeft; een n-gram dat niet in `theirs` staat, telt 0 keer. In de startcode begint `matched` op 0, en de loop neemt steeds één n-gram `gram`. `matched += x` telt `x` op bij `matched`, en `min` geeft het kleinste van de getallen die de functie krijgt: `min(7, 2)` is 2. Vervang de twee `...`.

        De check roept `clipped({"the": 7}, {"the": 2, "cat": 1, "is": 1, "on": 1, "mat": 1})` aan, en dat geeft 2: het voorbeeld van Papineni en collega's, met de eerste referentie. Daarna `clipped({"the": 1, "cat": 1, "sat": 1}, {"the": 2, "cat": 1, "is": 1, "on": 1, "mat": 1})`, en dat geeft 2, voor *the* en *cat*. En eigen gevallen: een lege dictionary, n-grams die niet in `theirs` staan, bigrams, en de n-grams van het uitgewerkte voorbeeld en van *i will the next bus take*.
        """),
        L.task("Tel de n-grams van een vertaling die in een referentie staan, elk hoogstens zo vaak als de referentie het heeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_clipped():
    box_clipped = L.code_box("clipped", lines=6)
    box_clipped
    return (box_clipped,)


@app.cell(hide_code=True)
def check_clipped(box_clipped):
    result_clipped = L.check("clipped", box_clipped.value)
    result_clipped
    return (result_clipped,)


@app.cell(hide_code=True)
def hints_clipped():
    L.hints(
        "clipped",
        [
            "De `if` vraagt met `in` of `gram` een key van `theirs` is, zoals in *Dictionaries*. Daaronder telt `matched +=` op hoe vaak `gram` telt: `mine[gram]` is hoe vaak de vertaling het heeft, `theirs[gram]` hoe vaak de referentie het hoogstens heeft, en `min` geeft het kleinste van twee getallen.",
            "Zonder de `if` stopt `theirs[gram]` met een `KeyError` bij een n-gram dat niet in de referentie staat, zoals *sat*. En neem de kleinste telling, niet de grootste: *the* 7 keer in de vertaling en twee keer in de referentie telt twee keer.",
            "`            matched += ___`",
        ],
        why="In de `if`: `gram in theirs`; daaronder: `matched += min(mine[gram], theirs[gram])`.",
    )
    return


@app.cell(hide_code=True)
def text_after_clipped():
    mo.md(r"""
    Woord voor woord, op de 21.636 zinnen van *Vertalen met regels*: 82.092 van de 118.401 woorden staan in een vertaling door mensen, 69,3%. Van de 96.765 bigrams zijn dat er 28.737, 29,7%; van de trigrams 15,5%; en van de groepjes van vier slechts 8,7%, 4.752 van 54.366. De meeste losse woorden kloppen, maar een langer groepje heeft ook de woorden in de goede volgorde nodig, en juist daar faalt woord voor woord.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bleu():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. bleu: van tellingen naar een score

        Schrijf `bleu`, dat `matched` en `total` krijgt, twee lijsten met de afgetopte telling en het aantal n-grams voor elke n, vanaf 1; `c`, de lengte van de vertaling in woorden; en `r`, de lengte van de referentie. Geef BLEU terug, een getal van 0 tot 1. De startcode geeft meteen 0 terug als een afgetopte telling 0 is, want dan is het meetkundig gemiddelde 0, en de logaritme van 0 bestaat niet. Daarna zet de startcode de logaritme van elke precision in de lijst `logs`, en geeft `penalty` keer *e* tot de macht `mean` terug. Vervang de twee `...`: `mean` is het gemiddelde van `logs`, en `penalty`, in de `else`, de brevity penalty van een vertaling die niet langer is dan de referentie.

        De check roept `bleu([6, 4, 2, 1], [6, 5, 4, 3], 6, 6)` aan, en dat geeft ongeveer 0,6043: het meetkundig gemiddelde van 6/6, 4/5, 2/4 en 1/3, zonder penalty. Daarna `bleu([4, 3, 2, 1], [4, 3, 2, 1], 4, 6)`, en dat geeft ongeveer 0,6065: elk n-gram telt, maar de vertaling heeft 4 woorden waar de referentie er 6 heeft. En eigen gevallen, waaronder de tellingen van de drie manieren van vertalen in het plaatje hieronder, opgeteld over alle zinnen.
        """),
        L.task("Bereken BLEU uit de afgetopte tellingen, de aantallen n-grams en de twee lengtes."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bleu():
    box_bleu = L.code_box("bleu", lines=13)
    box_bleu
    return (box_bleu,)


@app.cell(hide_code=True)
def check_bleu(box_bleu):
    result_bleu = L.check("bleu", box_bleu.value)
    result_bleu
    return (result_bleu,)


@app.cell(hide_code=True)
def hints_bleu():
    L.hints(
        "bleu",
        [
            "`np.mean` geeft het gemiddelde van een lijst, zoals in *De loss waarop je traint en de metric die je meldt*, en `np.exp(x)` geeft *e* tot de macht `x`, zoals in *Van score naar kans: de sigmoid*. In de `else` is `penalty` *e* tot de macht 1 min de lengte van de referentie gedeeld door de lengte van de vertaling.",
            "Zet `r` boven en `c` onder de breukstreep: een vertaling die korter is dan de referentie heeft `r / c` boven 1, dus de macht is kleiner dan 0 en de penalty kleiner dan 1. En neem het gemiddelde van de logaritmen, niet het totaal: vier logaritmen opgeteld zijn de logaritme van het product.",
            "`        penalty = np.exp(___)`",
        ],
        why="`mean = np.mean(logs)` en `penalty = np.exp(1 - r / c)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_bleu():
    mo.md(r"""
    Woord voor woord scoort 22,9 op de 21.636 zinnen. De 118.401 woorden zijn er iets minder dan de 118.658 van de dichtstbijzijnde referenties, dus de brevity penalty is 0,998. Met de regel van *Vertalen met regels* zakt BLEU naar 22,7, terwijl de regel 27 precieze vertalingen meer opleverde. De regel verplaatst alleen woorden, dus het aantal unigrams dat telt blijft gelijk, 82.092, en het aantal bigrams dat telt zakt van 28.737 naar 28.381. Van de 5.361 zinnen die de regel verandert, delen er nu 920 meer groepjes van twee tot vier woorden met een vertaling door mensen, en 1.293 minder. 595 van deze 1.293 zijn vragen, zoals *Kan ik helpen?*, dat *can help i* wordt. Twee manieren van meten, twee antwoorden: de score die je kiest, bepaalt wat als beter telt.

    Mensen tegen mensen scoren 57,8: ver boven woord voor woord, en ver onder 100. 787 van de 3.707 eerste vertalingen zijn precies gelijk aan een andere vertaling; de rest zegt hetzelfde met andere woorden. Zin voor zin is BLEU 0 bij 2.200 van deze 3.707, en bij 18.498 van de 21.636 vertalingen woord voor woord, 85,5%. Die nullen komen van de groepjes van vier: één $p_n$ van 0 maakt het meetkundig gemiddelde 0, en eerst de tellingen van alle zinnen optellen voorkomt dat. Ook de lengtes voor de brevity penalty tellen Papineni en collega's eerst op: wie de brevity penalty per zin berekent en de penalties middelt, straft afwijkingen in lengte bij korte zinnen erg hard af. Voor een losse zin past sacreBLEU smoothing toe: een n-gram-lengte zonder overeenkomst krijgt een kleine precision in plaats van 0. Daarmee krijgt *Can I help you?* tegen *May I help you?* 59,5 in plaats van 0.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chars():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. char_ngrams: tel groepjes letters

        Schrijf `char_ngrams`, dat `letters` krijgt, een tekst zonder spaties, en een lengte `n`. Geef een dictionary terug met elk groepje van `n` letters op een rij dat in `letters` voorkomt, en hoe vaak het voorkomt. In de startcode gaat de loop over de plaatsen `i` waar zo'n groepje begint, en telt de `if` het groepje `gram` in de dictionary `counts`; `counts[gram] += 1` telt 1 op bij de telling van `gram`. Vervang de twee `...`.

        De check roept `char_ngrams("banana", 2)` aan, en dat geeft `{"ba": 1, "an": 2, "na": 2}`, en `char_ngrams("hand", 3)`, en dat geeft `{"han": 1, "and": 1}`. Daarna eigen gevallen: een tekst korter dan `n`, een lege tekst, groepjes die overlappen, en de letters van Nederlandse zinnen met een samenstelling, zoals `C.letters(C.words("Alles hier is handgemaakt."))`: `C.letters` zet de woorden van een lijst aan elkaar, zonder spaties.
        """),
        L.task("Tel elk groepje van n letters op een rij in een tekst."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chars():
    box_chars = L.code_box("char_ngrams", lines=9)
    box_chars
    return (box_chars,)


@app.cell(hide_code=True)
def check_chars(box_chars):
    result_chars = L.check("char_ngrams", box_chars.value)
    result_chars
    return (result_chars,)


@app.cell(hide_code=True)
def hints_chars():
    L.hints(
        "char_ngrams",
        [
            "Een tekst van 6 letters heeft 5 bigrams: het laatste groepje van `n` begint op plaats `len(letters) - n`, en `range(k)` stopt net voor `k`: tot waar moet `range` dan gaan om plaats `len(letters) - n` te halen? Een slice `letters[a:b]` geeft de letters vanaf plaats `a` tot plaats `b`, zonder `b` zelf, zoals in *Lijsten*; `gram` is de slice van `n` letters die op plaats `i` begint.",
            "De slice stopt op plaats `i + n`, niet op `n`: `n` is een lengte, en de slice heeft een plaats nodig. En met `range(len(letters))` geven de laatste plaatsen groepjes die korter zijn dan `n`, zoals de *a* aan het eind van *banana*.",
            "`        gram = ___`",
        ],
        why="`for i in range(len(letters) - n + 1):` en `gram = letters[i:i + n]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_chars():
    mo.md(r"""
    Bij de vier Nederlandse paren geeft BLEU drie keer 0. *Tom is een rechtenstudent* deelt *tom is een*, drie woorden, met *Tom is een student rechten*, en geen groepje van vier. chrF geeft deze drie 71,7 tot 80,6, omdat de meeste groepjes letters nog overeenkomen: *rechten* en *student* zitten allebei in *rechtenstudent*. Alleen *Ik had vandaag een wiskundeproefwerk* deelt een groepje van vier woorden met de referentie, *ik had vandaag een*: BLEU 54,8, chrF 82,3. Volgens Popović houdt chrF deels rekening met woordvormen en zinsbouw, en werkt chrF voor elke taal, los van hoe je een tekst in woorden knipt. In 2016 schreef Popović dat het grootste voordeel van chrF is dat aanvaardbare variatie in goede vertalingen niet te zwaar telt.

    Op de vertalingen naar het Engels is chrF 51,1 voor woord voor woord, 50,9 met de regel en 71,4 voor mensen tegen mensen: dezelfde volgorde als bij BLEU, met de drie dichter bij elkaar.

    ### Probeer het: scoor een vertaling

    Typ een referentie en een vertaling, en zie de n-grams van woorden, de brevity penalty, BLEU, en de n-grams van tekens van chrF. Probeer *the cat is on* tegen *the cat is on the mat*, een vertaling waarin woorden ontbreken, of een Nederlandse zin met en zonder samenstelling, zoals *Ik heb een proefwerk wiskunde* en *Ik heb een wiskundeproefwerk*.
    """)
    return


@app.cell(hide_code=True)
def box_texts():
    try_reference = mo.ui.text(value="May I help you?", label="De referentie", full_width=True)
    try_translation = mo.ui.text(value="Can I help you?", label="De vertaling", full_width=True)
    mo.vstack([try_reference, try_translation])
    return try_reference, try_translation


@app.cell(hide_code=True)
def box_cell(try_reference, try_translation):
    mo.Html(C.box_picture(L.lang, try_reference.value, try_translation.value))
    return


@app.cell(hide_code=True)
def text_exercise_beta():
    mo.md(r"""
    ### Oefening 4. Wat heeft chrF liever?

    Neem een referentie en twee vertalingen. Vertaling A heeft elk n-gram van tekens van de referentie, en nog eens zoveel n-grams van tekens die niet in de referentie staan: precision 0,5, recall 1. Vertaling B heeft de helft van de n-grams van tekens van de referentie en verder niets: precision 1, recall 0,5.
    """)
    return


@app.cell(hide_code=True)
def choice_beta_cell():
    beta = L.choice(
        "beta",
        question="Welke van de twee krijgt de hoogste chrF, met β = 2?",
        options={
            "b": "B, die niets fout toevoegt",
            "right": "A, die niets mist",
            "same": "Gelijk: allebei de helft",
            "zero": "Geen van beide: allebei 0",
        },
        correct="right",
        why={
            "right": "A: (1 + 4) × 0,5 × 1 / (4 × 0,5 + 1) = 2,5 / 3, ongeveer 0,83. B: (1 + 4) × 1 × 0,5 / (4 × 1 + 0,5) = 2,5 / 4,5, ongeveer 0,56. Met β = 2 telt recall twee keer zo zwaar als precision: de helft van de referentie weglaten kost meer dan woorden toevoegen. Met β = 1 zouden beide 2 × 0,5 × 1 / 1,5 krijgen, ongeveer 0,67. Popović vond in 2016 dat recall nog belangrijker is bij talen met veel woordvormen.",
            "b": "B heeft niets fout, maar mist de helft van de referentie. Wat telt chrF zwaarder, precision of recall?",
            "same": "Met β = 1 zouden beide gelijk eindigen, op ongeveer 0,67. Welke van de twee telt zwaarder met β = 2?",
            "zero": "chrF is alleen 0 als geen enkel n-gram van tekens overeenkomt, en beide vertalingen hebben minstens de helft goed. Wat geeft de formule voor elk?",
        },
    )
    beta
    return (beta,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Vertalen met regels](/course/open/nlp-classic-mt?lang=nl) zei goed of fout; BLEU en chrF geven dezelfde vertalingen een deel van de punten, en oordelen over de regel anders dan het tellen van precieze overeenkomsten. De n-grams van [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl) zijn wat BLEU en chrF tellen, en de precision en recall van [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl) zijn wat chrF combineert. De volgende les, *Vertalers en machines samen*, kijkt naar mensen die verbeteren wat een machine vertaalde, en meet hoeveel er verandert.

    **Een voorbeeld uit het dagelijks leven.** Twee mensen wijzen de weg naar het station: *sla linksaf bij de kerk* en *ga bij de kerk naar links*. Een score die gedeelde woorden telt, geeft de tweede weinig punten tegen de eerste, terwijl je met allebei aankomt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_clipped, result_bleu, result_chars, beta):
    L.wrap_up(result_clipped, result_bleu, result_chars, beta)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
