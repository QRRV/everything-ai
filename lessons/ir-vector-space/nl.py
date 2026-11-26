# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ir-vector-space. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ir-vector-space", "nl")
    C = L.checks   # de dertien documenten van de course, de alinea's en vragen van SQuAD, de vectoren en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [De inverted index](/course/open/ir-inverted-index?lang=nl), en een over [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_order():
    quick_order = L.choice(
        "quick-order",
        question="In *De inverted index* gaf `search_and` de alinea's met alle woorden van een zoekvraag. In welke volgorde gaf de functie deze alinea's?",
        options={
            "best": "de alinea die het best bij de zoekvraag past eerst",
            "right": "op volgorde van nummer, van klein naar groot",
            "short": "de kortste alinea eerst",
            "most": "eerst de alinea waarin de woorden van de zoekvraag het vaakst voorkomen, dan de rest",
        },
        correct="right",
        why={
            "right": "De wandelingen bewaren de nummers in de volgorde van de postings lists: van klein naar groot. Deze volgorde zegt niets over hoe goed een alinea bij de zoekvraag past. Deze les rangschikt de alinea's, de beste eerst.",
            "best": "Om te weten welke alinea het best past, heeft de zoekmachine een score per alinea nodig. Staat zo'n score in een postings list? Zie [De inverted index](/course/open/ir-inverted-index?lang=nl).",
            "short": "Een postings list bevat de nummers van de alinea's, niets over de lengte. In welke volgorde staan deze nummers? Zie [De inverted index](/course/open/ir-inverted-index?lang=nl).",
            "most": "De index bewaart welke alinea's een woord bevatten, niet hoe vaak. In welke volgorde bewaart een wandeling de nummers? Zie [De inverted index](/course/open/ir-inverted-index?lang=nl).",
        },
    )
    quick_order
    return (quick_order,)


@app.cell(hide_code=True)
def quick_check_weight():
    quick_weight = L.choice(
        "quick-weight",
        question="In *TF-IDF: woorden wegen*: welke waarde krijgt een woord in een document als het woord in elk document van de verzameling voorkomt?",
        options={
            "count": "het aantal keer in het document",
            "right": "0, hoe vaak het er ook in voorkomt",
            "half": "de helft van het aantal: de IDF is 0,5",
            "most": "de hoogste waarde van het document, want zo'n woord komt het vaakst voor",
        },
        correct="right",
        why={
            "right": "De IDF is log10(N / df), en voor een woord in elk document is N / df gelijk aan 1, en de logaritme van 1 is 0. Een waarde 0 telt nergens mee in een score, ook niet in deze les.",
            "count": "Het aantal is de term frequency. Waarmee wordt het aantal vermenigvuldigd, en wat is dat getal voor een woord in elk document? Zie [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl).",
            "half": "De IDF is log10(N / df). Wat is N / df voor een woord in elk document, en wat is de logaritme daarvan? Zie [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl).",
            "most": "In *TF-IDF: woorden wegen* komt *the* 158 keer voor in *A Mad Tea-Party*. Welke waarde kreeg het woord daar? Zie [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl).",
        },
    )
    quick_weight
    return (quick_weight,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    XQuAD, een selectie uit SQuAD van Mikel Artetxe, Sebastian Ruder en Dani Yogatama, heeft vragen bij 240 van de 2.067 alinea's van de les [De inverted index](/course/open/ir-inverted-index?lang=nl), en SQuAD-NL bevat 1.183 van deze vragen in het Nederlands. Elke vraag is door een mens geschreven bij één alinea: de eigen alinea van de vraag. Andere alinea's kunnen ook helpen bij een vraag, maar deze les telt alleen de eigen alinea. Stel elke vraag aan de inverted index, met AND: de index vindt iets bij 12 van de 1.183 vragen, en de eigen alinea bij 10. Een vraag bevat bijna altijd een woord dat niet in de eigen alinea staat: de vraag over het parlement van Victoria hieronder bevat *welk*, *document* en *vormde*, en de eigen alinea geen van de drie. En ook als AND alinea's vindt, komen de alinea's op volgorde van nummer, niet op volgorde van hoe goed elke alinea past.

    [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl) gaf een document als score de waarden van de zoekwoorden erin, bij elkaar opgeteld. Deze score rangschikt elke alinea voor elke vraag, en de eigen alinea komt bovenaan bij 44,7% van de vragen. Maar kijk waar de eerste plaatsen terechtkomen. Verdeel de alinea's op lengte in vijf groepen van ongeveer dezelfde grootte: de langste groep, van 160 tot 659 woorden, bevat de eigen alinea van 24,9% van de vragen, en krijgt 52,9% van de eerste plaatsen. Bij de vraag *Welk document vormde het parlement van Victoria?* staat bovenaan een alinea van 567 woorden over het recht van de Europese Unie: *parlement* staat er twaalf keer in, en *Victoria* geen enkele keer. De eigen alinea, 97 woorden met het antwoord, de Victoria Constitution Act 1855, komt op plaats vier. Een lange alinea bevat meer verschillende woorden en herhaalt woorden vaker, en verzamelt zo meer waarde, waar de alinea ook over gaat.

    Deze les ziet een vraag en een alinea als twee vectoren, en vergelijkt de richting waarin de twee vectoren wijzen, niet hoeveel waarde erin zit. Dat is het vector space model, een van de klassieke manieren om documenten te rangschikken in information retrieval.

    **Een voorbeeld uit het dagelijks leven.** Een grote pan soep en een klein kommetje van dezelfde soep smaken hetzelfde: de hoeveelheid verschilt, de verhouding niet. De cosine similarity vergelijkt de verhouding van de woorden, niet de hoeveelheid.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_cosine = L.guess(
        "guess",
        question="Met de TF-IDF-score komt de eigen alinea bovenaan bij 44,7% van de 1.183 vragen. Maak nu ook van elke vraag een vector van TF-IDF-waarden, en rangschik de alinea's op de cosine similarity van de twee vectoren, uit *Lengte, afstand en cosine similarity*. Bij welk deel van de vragen komt de eigen alinea dan bovenaan?",
        options={
            "same": "ongeveer even vaak als eerst, want delen door een lengte verandert geen volgorde",
            "fewer": "minder vaak: ongeveer een op de drie",
            "right": "ongeveer twee op de drie",
            "most": "ongeveer negen op de tien",
        },
        reveal="Ongeveer twee op de drie: 62,6%. Met de cosine similarity krijgt de langste groep 22,7% van de eerste plaatsen, en hebben de eerste alinea's gemiddeld 130,3 woorden, tegen 198,6 met de TF-IDF-score; de eigen alinea's hebben er gemiddeld 132,5. Een deel van de winst komt van de vraag: alleen al het wegen van de woorden van de vraag met de IDF brengt het deel op 52,8%, nog zonder te delen.",
    )
    guess_cosine
    return (guess_cosine,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Een document als vector van waarden.** In [Een document als vector](/course/open/nlp-document-vectors?lang=nl) werd een document een vector van aantallen, één getal per woord. Weeg elk aantal met de IDF, en elk van de 2.067 alinea's wordt een vector van 30.090 getallen, één voor elk verschillend woord van de alinea's, bijna allemaal 0. Christopher Manning, Prabhakar Raghavan en Hinrich Schütze noemen dit in *Introduction to Information Retrieval* (2008) het vector space model: een verzameling documenten als vectoren in één gedeelde ruimte. Het boek wijst op het SMART-systeem van Gerard Salton, beschreven in 1971, als mogelijk het eerste systeem dat een document zag als een vector van waarden.

    **Ook de zoekvraag is een vector.** Volgens het boek kun je ook een zoekvraag zien als een vector. De vraag *Welk document vormde het parlement van Victoria?* krijgt een waarde voor elk woord, het aantal keer maal de IDF onder de alinea's: *document* 2,71, *welk* 2,41, *vormde* 2,01, *victoria* 1,81, *parlement* 1,58, *het*, dat in 1.898 van de 2.067 alinea's staat, 0,037, en *van* 0,01. Een woord van de zoekvraag dat in geen enkele alinea staat, heeft geen IDF en valt weg.

    **Het inproduct.** Vermenigvuldig de waarden van elk woord dat in beide vectoren staat, en tel de producten op: het inproduct uit [Veel features: het inproduct](/course/open/math-dot-product?lang=nl). De TF-IDF-score van *TF-IDF: woorden wegen* is ook een inproduct, met als vector van de zoekvraag het aantal keer dat elk woord erin staat, zonder IDF. Weeg je ook de woorden van de zoekvraag met de IDF, dan telt een zeldzaam woord van de vraag zwaarder mee: de eigen alinea komt dan bovenaan bij 52,8% van de vragen. Een woord dat niet in de zoekvraag staat, voegt niets toe aan het inproduct, dus alleen de postings lists van de woorden van de zoekvraag doen ertoe: een index met een waarde naast elke posting geeft elk inproduct zonder een alinea te openen.

    **De cosine similarity.** Het inproduct groeit nog steeds met de lengte van een alinea: schrijf een alinea twee keer achter elkaar, en elke waarde verdubbelt, en het inproduct met elke zoekvraag ook. Om bij het vergelijken van twee documenten de invloed van de lengte weg te nemen, gebruikt het boek de cosine similarity uit [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl). De cosine similarity deelt het inproduct door de lengtes van beide vectoren:

    $$\cos(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\lVert \mathbf{q} \rVert \, \lVert \mathbf{d} \rVert}$$

    In de noemer staat volgens het boek het product van de twee euclidische lengtes: de norm van elke vector, de wortel uit de kwadraten bij elkaar opgeteld. Delen door $\lVert \mathbf{d} \rVert$ neemt het voordeel van een lange alinea weg. De lengte van de vector van de zoekvraag is voor elke alinea gelijk, dus verandert geen volgorde; door de deling door beide lengtes is de cosine similarity hooguit 1, en omdat geen enkele waarde kleiner is dan 0, ook nooit kleiner dan 0. Dit is **length normalization**: een score gedeeld door de lengte van de vector van het document.

    **Pivoted length normalization.** Amit Singhal, Chris Buckley en Mandar Mitra (1996) verdeelden documenten op lengte in groepen, en vergeleken per groep hoe vaak de documenten relevant waren met hoe vaak de cosine similarity de documenten vond. Lange documenten werden minder vaak gevonden dan op grond van de relevantie te verwachten was. Het boek vat het zo samen: de normalisatie van de cosine similarity vertekent de berekende relevantie ten opzichte van de echte relevantie, ten koste van langere documenten. De oplossing deelt het inproduct door een mengsel van de lengte van het document en een vast getal, de **pivot**:

    $$\text{helling} \cdot \lVert \mathbf{d} \rVert + (1 - \text{helling}) \cdot \text{pivot}$$

    met een **helling** kleiner dan 1. Als pivot kiest het boek de lengte waarbij de kromme van hoe vaak documenten relevant zijn de kromme snijdt van hoe vaak de cosine similarity de documenten vindt. Een document met een vector langer dan de pivot krijgt een deler kleiner dan de eigen lengte, dus een hogere score dan wanneer het inproduct door de eigen lengte gedeeld wordt, zoals bij de cosine similarity; een korter document krijgt een grotere deler. Helling 1 geeft de volgorde van de cosine similarity, en helling 0 deelt elk document door hetzelfde getal, en houdt zo de volgorde van het inproduct.

    De alinea's van deze les laten het patroon van Singhal en collega's zien: met de cosine similarity krijgt de langste groep 22,7% van de eerste plaatsen, terwijl de groep de eigen alinea van 24,9% van de vragen bevat, en staat de eigen alinea in de langste groep, dan komt de alinea bovenaan bij 54,2% van de vragen, tegen 65,4% bij de andere vragen. Singhal en collega's kozen als pivot het gemiddelde van de oude delers, en deze les doet hetzelfde: met de gemiddelde lengte van de vectoren van de alinea's, 23,2, als pivot en helling 0,8 krijgt de langste groep 26,0% van de eerste plaatsen, en de vragen van deze groep 57,3%; de andere vragen zakken naar 63,3%, en alle vragen samen naar 61,8%. Op deze alinea's verschuift de pivot eerste plaatsen van korte naar lange alinea's, maar komt de eigen alinea daardoor niet vaker bovenaan. Het boek waarschuwt dat pivoted length normalization niet bij elke toepassing past.

    Het plaatje laat de vraag over het parlement van Victoria zien in de drie rangschikkingen, en waar de eerste plaatsen van alle vragen terechtkomen.
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
    | **vector space model** | documenten en zoekvragen als vectoren van waarden, met één dimensie per woord, gerangschikt op hoe dicht de vector van een document bij de vector van de zoekvraag ligt |
    | **vector van de zoekvraag** | de zoekvraag als vector van waarden, in dezelfde ruimte als de documenten: het aantal keer van elk woord van de zoekvraag maal de IDF |
    | **length normalization** | een score gedeeld door de lengte van de vector van het document, zodat een lang document niet wint door de lengte alleen; de cosine similarity doet dit |
    | **pivoted length normalization** | delen door een mengsel van de lengte van de vector van het document en een vaste pivot, zodat lange documenten minder verliezen dan met de cosine similarity |
    | **pivot**, **helling** | het vaste getal van pivoted length normalization, en het deel van de lengte in het mengsel: helling 1 geeft de volgorde van de cosine similarity |
    | $\mathbf{q} \cdot \mathbf{d}$ | het inproduct van de vector $\mathbf{q}$ van de zoekvraag en de vector $\mathbf{d}$ van het document |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om documenten te rangschikken voor een zoekvraag met het vector space model:

    1. Maak van elk document een vector: het aantal keer van elk woord maal de IDF.
    2. Maak op dezelfde manier een vector van de zoekvraag, met de IDF van de documenten; laat een woord weg dat in geen enkel document staat.
    3. Bereken het inproduct van de vector van de zoekvraag met de vector van elk document. Alleen de woorden van de zoekvraag tellen mee, dus de postings lists van deze woorden zijn genoeg.
    4. Deel elk inproduct door de lengte van de vector van de zoekvraag en de lengte van de vector van het document: de cosine similarity.
    5. Sorteer de documenten op cosine similarity, de hoogste eerst.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.SMALL_IDF` bevat de IDF van elk woord van de dertien documenten van de course, en `C.SMALL_VECTORS` de TF-IDF-vectoren van de documenten, elk een dictionary van woorden naar waarden. `C.vector` maakt van een lijst woorden een vector met een gegeven IDF, `C.dot_product` geeft het inproduct van twee vectoren, `C.norm` de lengte van een vector, en `C.cosine` de cosine similarity van twee vectoren. De zoekvraag is *rode wijn hart*:
    """)
    return


@app.cell
def example_cell():
    _idf = C.SMALL_IDF[L.lang]
    _query = C.vector(C.QUERIES[L.lang]["wine"], _idf)
    print({_word: round(_weight, 3) for _word, _weight in _query.items()})
    for _doc in (6, 7, 8):
        _vector = C.SMALL_VECTORS[L.lang][_doc]
        _dot = C.dot_product(_query, _vector)
        print(_doc, round(_dot, 3), round(C.norm(_vector), 3), round(C.cosine(_query, _vector), 3))
    _twice = C.vector(C.DOCS[L.lang][6] * 2, _idf)
    print(round(C.dot_product(_query, _twice), 3), round(C.norm(_twice), 3), round(C.cosine(_query, _twice), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *Rode* en *wijn* staan in vier van de dertien documenten en krijgen 0,512; *hart* staat in drie en krijgt 0,637. Elke regel geeft een document, het inproduct met de vector van de zoekvraag, de lengte van de vector en de cosine similarity. De documenten 7 en 8 bevatten *rode*, *wijn* en *hart* elk één keer, dus het inproduct is gelijk: 0,93. Document 8 bevat daarnaast meer woorden, dus de vector is langer, 3,479 tegen 3,24, en de cosine similarity lager: 0,277 tegen 0,298. De laatste regel is document 6, twee keer achter elkaar: het inproduct verdubbelt, van 1,454 naar 2,907, de lengte verdubbelt, van 3,905 naar 7,81, en de cosine similarity blijft 0,386.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_norm():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. norm: de lengte van een vector van waarden

        Schrijf `norm`, die `weights` krijgt, een vector als dictionary van woorden naar waarden, zoals `C.SMALL_VECTORS["nl"][6]`, de vector van document 6. De functie geeft de lengte van de vector terug: de wortel uit de kwadraten van de waarden bij elkaar opgeteld, de norm $\lVert \mathbf{x} \rVert$ uit *Lengte, afstand en cosine similarity*. Een woord dat niet in de dictionary staat, heeft waarde 0 en voegt niets toe, dus alleen de woorden in de dictionary tellen mee. In de startcode begint `total` bij 0, gaat de loop elk `word` van `weights` langs, en neemt `** 0.5` aan het eind de wortel. Vervang de `...` door de nieuwe `total`.

        De check roept `norm({"red": 3.0, "wine": 4.0})` aan, wat `5.0` geeft, en `norm({"python": 1.0, "snake": 2.0, "tank": 2.0})`, wat `3.0` geeft. Daarna eigen gevallen: een lege vector, een waarde 0, vier waarden 1, één enkele waarde, en de vectoren van document 6 in de Engelse les en document 10 in deze les.
        """),
        L.task("Geef de lengte van een vector als dictionary: de wortel uit de kwadraten van de waarden bij elkaar opgeteld."),
    ])
    return


@app.cell(hide_code=True)
def exercise_norm():
    box_norm = L.code_box("norm", lines=6)
    box_norm
    return (box_norm,)


@app.cell(hide_code=True)
def check_norm(box_norm):
    result_norm = L.check("norm", box_norm.value)
    result_norm
    return (result_norm,)


@app.cell(hide_code=True)
def hints_norm():
    L.hints(
        "norm",
        [
            "Elke ronde telt één waarde op bij `total`. `weights[word]` is de waarde van `word`, en de norm telt de kwadraten van de waarden op, niet de waarden zelf.",
            "Het nieuwe totaal is de oude `total` plus het kwadraat van de waarde van dit woord; in Python neemt `** 2` het kwadraat van een getal.",
            "`        total = total + ___`",
        ],
        why="`total = total + weights[word] ** 2`: elk woord telt het kwadraat van de waarde op, en de wortel uit het totaal is de lengte.",
    )
    return


@app.cell(hide_code=True)
def text_after_norm():
    mo.md(r"""
    In het plaatje is de vector van een alinea in de langste groep ongeveer twee keer zo lang als een vector in de kortste: gemiddeld 31,95 tegen 16,65. Een lange alinea bevat meer verschillende woorden en herhaalt woorden, en elk woord maakt de vector langer. De cosine similarity deelt door deze lengte.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cosine():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. cosine: het inproduct gedeeld door beide lengtes

        Schrijf `cosine`. De functie krijgt twee vectoren als dictionary, `q`, de vector van de zoekvraag, en `d`, de vector van een document, en geeft de cosine similarity van de twee terug. In de startcode geeft de `if` het getal `0.0` als een van de twee vectoren lengte 0 heeft, zoals bij een zoekvraag met woorden uit geen enkel document: de cosine similarity bestaat dan niet, en de functie geeft in plaats daarvan 0.0. De loop gaat elk `word` van `q` langs, en als `d` het woord ook bevat, telt de loop het product van de twee waarden, `q[word] * d[word]`, op bij `dot`: het inproduct. Vervang de `...` door de cosine similarity. Het vak kent de functie `norm` van oefening 1 al.

        De check roept `cosine({"red": 1.0, "wine": 1.0}, {"red": 2.0, "wine": 2.0})` aan, twee vectoren die dezelfde kant op wijzen, wat `1.0` geeft, en `cosine({"wine": 3.0, "heart": 4.0}, {"wine": 1.0})`, wat `0.6` geeft. Daarna eigen gevallen: vectoren zonder gemeenschappelijk woord, een lege zoekvraag, een zoekvraag met als enige waarde 0, een document met een woord dat niet in de zoekvraag staat, twee gelijke vectoren, en de zoekvraag *red wine heart* met de documenten 6 en 8 van de Engelse les, en *rode wijn hart* met document 7 van deze les.
        """),
        L.task("Geef de cosine similarity van twee vectoren: het inproduct gedeeld door de lengte van elk van de twee."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cosine():
    box_cosine = L.code_box("cosine", lines=9)
    box_cosine
    return (box_cosine,)


@app.cell(hide_code=True)
def check_cosine(box_cosine):
    result_cosine = L.check("cosine", box_cosine.value)
    result_cosine
    return (result_cosine,)


@app.cell(hide_code=True)
def hints_cosine():
    L.hints(
        "cosine",
        [
            "`dot` bevat het inproduct, en `norm` geeft de lengte van een vector. De cosine similarity deelt `dot` door de lengte van `q` en de lengte van `d`, met elkaar vermenigvuldigd.",
            "Deel het inproduct door één getal: de lengte van de vector van de zoekvraag maal de lengte van de vector van het document, tussen haakjes, zodat Python door beide tegelijk deelt.",
            "`    return dot / (___)`",
        ],
        why="`return dot / (norm(q) * norm(d))`: het inproduct gedeeld door het product van de twee lengtes.",
    )
    return


@app.cell(hide_code=True)
def text_after_cosine():
    mo.md(r"""
    Bij de vraag over het parlement van Victoria heeft de alinea over het recht van de Europese Unie het hoogste inproduct, 30,1, tegen 17,4 voor de eigen alinea. Maar de vector van deze alinea is 52,6 lang, en de vector van de eigen alinea 18,5: 30,1 gedeeld door 52,6 is kleiner dan 17,4 gedeeld door 18,5, en de cosine similarity zet de eigen alinea bovenaan. Bij de vraag over de motor van Energiprojekt AB zet de TF-IDF-score een alinea van 197 woorden bovenaan, en de cosine similarity de eigen alinea, van 85 woorden.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pivoted():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. pivoted: delen door een mengsel van de lengte en de pivot

        Schrijf `pivoted`, die `dot` krijgt, het inproduct van de vector van een zoekvraag en de vector van een document, `length`, de lengte van de vector van het document, `pivot`, het vaste getal uit *Het idee*, en `slope`, de helling, een getal van 0 tot 1. De functie geeft het inproduct terug, gedeeld door de deler van Singhal en collega's, waarin de lengte meetelt met het deel `slope` en de pivot met het deel `1 - slope`. De startcode noemt dit getal `divisor`, en deelt `dot` erdoor. Vervang de `...` door de deler. Met alleen `+`, `-`, `*` en `/` werkt de functie ook als `dot` en `length` NumPy-arrays zijn: het plaatje hieronder roept de functie aan met alle vragen en alle 2.067 alinea's tegelijk.

        De check roept `pivoted(5.6, 30.0, 20.0, 0.8)` aan, een vector van een document langer dan de pivot, wat `0.2` geeft, en `pivoted(3.0, 10.0, 20.0, 0.8)`, een kortere vector, wat `0.25` geeft. Daarna eigen gevallen: helling 1, helling 0, een lengte gelijk aan de pivot, en NumPy-arrays met één rij en met twee rijen.
        """),
        L.task("Deel het inproduct door een mengsel van de lengte en de pivot, waarin de lengte het deel slope krijgt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pivoted():
    box_pivoted = L.code_box("pivoted", lines=4)
    box_pivoted
    return (box_pivoted,)


@app.cell(hide_code=True)
def check_pivoted(box_pivoted):
    result_pivoted = L.check("pivoted", box_pivoted.value)
    result_pivoted
    return (result_pivoted,)


@app.cell(hide_code=True)
def hints_pivoted():
    L.hints(
        "pivoted",
        [
            "`divisor` is een mengsel van twee getallen: `pivot` telt mee met het deel `(1 - slope)`, en de lengte van het document met het deel `slope`.",
            "Het tweede deel van het mengsel is de lengte van de vector van het document, vermenigvuldigd met het eigen deel, de helling.",
            "`    divisor = (1 - slope) * pivot + ___`",
        ],
        why="`divisor = (1 - slope) * pivot + slope * length`: bij helling 1 is de deler de lengte, zoals bij de cosine similarity; bij helling 0 is de deler voor elk document de pivot.",
    )
    return


@app.cell(hide_code=True)
def text_after_pivoted():
    mo.md(r"""
    In de voorbeelden van de check krijgt het document met lengte 30 de deler 28: een score van 0,2, waar delen door de lengte 0,187 zou geven. Het document met lengte 10 krijgt de deler 12: 0,25, waar de lengte 0,3 zou geven. In het plaatje geeft helling 0,8 de langste groep alinea's 26,0% van de eerste plaatsen: iets meer dan de 24,9% van de vragen met de eigen alinea in de groep, waar de cosine similarity 22,7% gaf, en toch komt de eigen alinea bij minder vragen bovenaan dan met de cosine similarity.

    ### Probeer het: de helling

    Verschuif de helling. Bij 1 geeft de helling de cosine similarity; bij 0 de volgorde van het inproduct. Het plaatje rangschikt de 2.067 alinea's voor alle vragen met deze helling, en laat zien hoe vaak de eigen alinea bovenaan komt, en waar de eerste plaatsen terechtkomen. Zoek de helling waarbij het deel van de eerste plaatsen van de langste groep het dichtst bij het deel van de eigen alinea's komt. Ga dan naar 0, en kijk hoe de lange alinea's de eerste plaatsen weer innemen.
    """)
    return


@app.cell(hide_code=True)
def try_slide():
    try_slope = mo.ui.slider(0.0, 1.0, step=0.05, value=C.TRY_START, label="Helling", show_value=True)
    try_slope
    return (try_slope,)


@app.cell(hide_code=True)
def try_look(try_slope):
    mo.Html(C.try_picture(L.lang, try_slope.value))
    return


@app.cell(hide_code=True)
def text_exercise_changes():
    mo.md(r"""
    ### Oefening 4. Wat verandert de cosine similarity?

    Neem de zoekvraag *rode wijn hart* en document 6 van de course, *Rode wijn, witte wijn en het hart: maakt de kleur van de wijn verschil voor de kans op een hartaanval? Wat onderzoek vond.* Deel elke verandering hieronder in: de cosine similarity van de vector van de zoekvraag en de vector van document 6 blijft gelijk, of verandert. De IDF van elk woord verandert niet.
    """)
    return


@app.cell(hide_code=True)
def exercise_changes():
    sort_changes = L.sort("changes")
    sort_changes
    return (sort_changes,)


@app.cell(hide_code=True)
def check_changes(sort_changes):
    result_changes = L.check("changes", sort_changes)
    result_changes
    return (result_changes,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De waarden komen uit [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl), en de cosine similarity uit [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl), nu met vectoren als dictionary, omdat bijna elke waarde 0 is. Alleen de woorden van de zoekvraag tellen mee in een inproduct, dus de index uit [De inverted index](/course/open/ir-inverted-index?lang=nl), met een waarde naast elke posting, geeft elke score uit de postings lists van de zoekvraag. De volgende les, *BM25: de klassieke rangschikking*, gebruikt ook een IDF, en verandert twee dingen: elke herhaling van een woord telt minder mee dan de vorige, en een eigen instelling bepaalt hoe zwaar de lengte van een document meetelt.

    **Een voorbeeld uit het dagelijks leven.** Een foto afgedrukt als postzegel en dezelfde foto afgedrukt als poster tonen hetzelfde beeld: alleen de grootte verschilt. De cosine similarity ziet het beeld, niet de grootte.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_norm, result_cosine, result_pivoted, result_changes):
    L.wrap_up(result_norm, result_cosine, result_pivoted, result_changes)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
