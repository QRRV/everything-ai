# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-document-vectors. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-document-vectors", "nl")
    C = L.checks   # de woorden van de course, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [Spam of niet: naive Bayes](/course/open/lm-naive-bayes?lang=nl), en een over [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bag():
    quick_bag = L.choice(
        "quick-bag",
        question="Naive Bayes bekijkt een tekst als een bag of words: welke woorden erin staan, en hoe vaak. Bij welke verandering blijft de bag of words van een tekst hetzelfde?",
        options={
            "add": "Er komt één woord bij",
            "right": "De woorden komen in een andere volgorde",
            "twice": "Een van de woorden staat er nog een keer",
        },
        correct="right",
        why={
            "right": "Een bag of words houdt elk woord bij met het aantal, en vergeet waar het woord stond: *hond bijt man* en *man bijt hond* geven dezelfde bag. Deze les maakt van zo'n bag een lijst getallen met een vaste lengte, en neemt die blinde vlek dus over.",
            "add": "Een extra woord komt in de bag, of het aantal van dat woord gaat 1 omhoog. Zie [Spam of niet: naive Bayes](/course/open/lm-naive-bayes?lang=nl).",
            "twice": "Het aantal van dat woord gaat 1 omhoog, en de bag houdt aantallen bij. Zie [Spam of niet: naive Bayes](/course/open/lm-naive-bayes?lang=nl).",
        },
    )
    quick_bag
    return (quick_bag,)


@app.cell(hide_code=True)
def quick_check_cosine():
    quick_cosine = L.choice(
        "quick-cosine",
        question="Wat is de cosine similarity van `[1, 2]` en `[2, 4]`?",
        options={
            "zero": "0, want de vectoren verschillen",
            "half": "0,5, want de ene is de helft van de andere",
            "right": "1, want beide wijzen dezelfde kant op",
            "two": "2, de factor ertussen",
        },
        correct="right",
        why={
            "right": "`[2, 4]` is `[1, 2]` keer 2: twee keer zo lang, in dezelfde richting, en de cosine similarity kijkt alleen naar de richting. Het inproduct is 10, de lengtes zijn √5 en √20, en 10 / (√5 × √20) = 10 / 10 = 1. Deze les gebruikt de cosine similarity om teksten van verschillende lengte te vergelijken.",
            "zero": "Een cosine similarity van 0 betekent dat de vectoren haaks op elkaar staan, zoals `[1, 0]` en `[0, 1]`. Deze twee wijzen dezelfde kant op. Zie [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl).",
            "half": "De cosine similarity deelt het inproduct door beide lengtes, dus een factor tussen de vectoren valt weg. Zie [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl).",
            "two": "De cosine similarity ligt tussen −1 en 1, wat de lengtes ook zijn. Zie [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl).",
        },
    )
    quick_cosine
    return (quick_cosine,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een tekst kan drie woorden hebben of drieduizend, maar de meeste methodes die dingen vergelijken of indelen, willen elk ding als een lijst getallen van één vaste lengte. Deze les geeft een tekst zo'n lijst: één aantal voor elk woord van een vaste lijst woorden. Jurafsky en Martin beschrijven in *Speech and Language Processing* hoe het vector space model uit de information retrieval een document voorstelt als een vector met de aantallen van de woorden die erin staan, en noemen daarbij Salton, 1971. Information retrieval is het vakgebied van het zoeken, en vector space model is de naam die dat vakgebied gebruikt voor deze manier om documenten in vectoren om te zetten; het idee komt daarvandaan. Volgens het leerboek *Introduction to Information Retrieval* van Christopher Manning, Prabhakar Raghavan en Hinrich Schütze was het SMART-systeem van Gerard Salton en collega's aan Cornell misschien het eerste dat een document als een vector van getallen zag, één per woord. Een paper van Salton, A. Wong en C. S. Yang uit 1975 draagt de naam in de titel: *A vector space model for automatic indexing*. Met zulke lijsten kun je twee teksten vergelijken met de cosine similarity, en kan een nieuwe tekst het label krijgen van de tekst waar de nieuwe tekst het meest op lijkt.

    De teksten van deze les zijn 3.000 korte zinnen uit reviews, verzameld door Dimitrios Kotzias en collega's voor een paper in 2015 en gepubliceerd door de UCI Machine Learning Repository als *Sentiment Labelled Sentences*: 1.000 van Amazon, over producten zoals telefoons en headsets, 1.000 van IMDb, over films, en 1.000 van Yelp, over restaurants. Elke site levert 500 positieve en 500 negatieve zinnen; de makers probeerden zinnen te kiezen met een duidelijk positieve of negatieve toon. De zinnen zijn Engels. De les laat de 85 zinnen met scheldwoorden, beledigingen, scheldnamen, of grove of seksuele opmerkingen nooit zien, maar telt deze zinnen mee zoals de rest.

    **Een voorbeeld uit het dagelijks leven.** Het bestelformulier van een bakker noemt elk brood dat de bakker bakt, elk met een vakje voor hoeveel. Elke klant vult hetzelfde formulier in, dus je kunt elke twee bestellingen vakje voor vakje vergelijken, en de meeste vakjes blijven leeg.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="De les houdt 2.400 van de zinnen apart om mee te trainen. Samen gebruiken die zinnen 4.518 verschillende woorden. Schrijf elke zin als een lijst van 4.518 aantallen, één voor elk woord. Welk deel van alle getallen in die 2.400 lijsten is 0?",
        options={
            "half": "Ongeveer de helft",
            "ninety": "Ongeveer 90%",
            "right": "Ongeveer 99,8%",
            "more": "Ongeveer 99,99%",
        },
        reveal="**Ongeveer 99,8%**: van de 10.843.200 getallen zijn er maar 26.188 geen 0, dus 99,76% is 0. Een trainingszin heeft gemiddeld 10,9 verschillende woorden en hoogstens 56, van de 4.518. Twee zinnen hebben helemaal geen woord: het zijn allebei *10/10*, en de woorden van deze les zijn reeksen letters. 2.625 van de 4.518 woorden komen in alle 2.400 zinnen samen maar één keer voor.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Een vocabulary en een vector

    De **vocabulary** is een vaste lijst woorden, elk met een eigen nummer: de kolom. In deze les is dat elk woord uit de 2.400 trainingszinnen, in kleine letters en op alfabet: *a* is kolom 0, *good* kolom 1.707, *the* kolom 3.948. Een **documentvector** heeft één aantal voor elk woord van de vocabulary: hoe vaak dat woord in het document voorkomt. Een document is hier één zin. Jurafsky en Martin noemen dit een bag of words: je behandelt een tekst als een zak woorden zonder volgorde, waarin de plaats van elk woord verdwijnt en alleen telt hoe vaak het voorkomt. Manning en collega's geven een voorbeeld: in deze opvatting is het document *Mary is sneller dan John* hetzelfde als *John is sneller dan Mary*.

    ### De document-term matrix

    Zet de vectoren van alle documenten onder elkaar, en je krijgt een tabel met één rij per document en één kolom per woord: de **document-term matrix**. De documentatie van scikit-learn, een Python-library voor machine learning, beschrijft een verzameling documenten zo: als een matrix met één rij per document en één kolom per woord dat in de verzameling voorkomt. Beide leerboeken tekenen de matrix andersom, met een rij per woord en een kolom per document, en spreken dan van een *term-document matrix*. De afbeelding toont de matrix van vier zinnen, over alleen de twaalf woorden die erin staan.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Vooral nullen: sparse

    Een zin gebruikt een handvol van de 4.518 woorden, dus bijna elk getal in de vector is 0. Volgens de documentatie van scikit-learn is in zo'n matrix meestal meer dan 99% van de getallen 0. Een vector of matrix met zo veel nullen heet **sparse**. Al die nullen opslaan verspilt geheugen, dus een **sparse matrix** bewaart alleen de getallen die geen 0 zijn, elk met de eigen plaats. De documentatie van SciPy, de library met de sparse matrices die scikit-learn teruggeeft, zegt het zo: alleen de plaatsen waar data staat, worden vastgelegd. In gewoon Python doet een dictionary van kolom naar aantal hetzelfde: *The mic is great.* wordt `{1731: 1, 2086: 1, 2475: 1, 3948: 1}`, vier getallen in plaats van 4.518. Deze les noemt zo'n dictionary een **sparse rij**.

    ### Vergelijken met de cosine similarity

    Twee zinnen lijken op elkaar als de vectoren ongeveer dezelfde kant op wijzen. Volgens Jurafsky en Martin is de cosinus van de hoek tussen de vectoren verreweg de meest gebruikte maat voor gelijkenis. Het inproduct alleen volstaat niet: het bevoordeelt lange vectoren. Manning en collega's zeggen hetzelfde over documenten: twee documenten met bijna dezelfde inhoud kunnen ver uit elkaar liggen, alleen omdat het ene document veel langer is dan het andere. De cosine similarity deelt de lengte eruit. Aantallen zijn nooit negatief, en zoals Jurafsky en Martin opmerken, ligt de cosine similarity van zulke vectoren dan tussen 0 en 1: 1 als twee zinnen dezelfde woorden in dezelfde verhoudingen gebruiken, 0 als de zinnen geen woord delen.

    Met sparse rijen heeft het inproduct alleen de kolommen nodig die beide rijen hebben: in elke andere kolom is een van de twee aantallen 0, en het product dus ook.

    ### Een nieuwe zin krijgt het label van de meest gelijkende zin

    Deze les geeft een nieuwe zin een label op de eenvoudigste manier die de vectoren toelaten: zoek de trainingszin met de hoogste cosine similarity, en geef de nieuwe zin het label van die trainingszin. Om dat uit te proberen, houdt de les elke vijfde zin achter, 600 in totaal: die zinnen tellen niet mee voor de vocabulary en niet voor de trainingsrijen. Een woord van een nieuwe zin dat niet in de vocabulary staat, heeft geen kolom, en valt weg. scikit-learn doet hetzelfde: woorden die niet in de trainingsteksten stonden, worden volledig genegeerd.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **vocabulary** | een vaste lijst woorden, elk met een eigen kolom: hier elk woord uit de trainingszinnen, op alfabet |
        | **documentvector** | een lijst met één aantal voor elk woord van de vocabulary: hoe vaak het woord in het document voorkomt |
        | **document-term matrix** | de documentvectoren onder elkaar: één rij per document, één kolom per woord |
        | **sparse** | met vooral nullen |
        | **sparse matrix** | een matrix waarvan alleen de getallen die geen 0 zijn bewaard worden, elk met de eigen plaats |
        | **sparse rij** | in deze les: een dictionary van kolom naar aantal, met alleen de aantallen die geen 0 zijn |
        """),
        L.watch_out("Een hoge cosine similarity betekent dat twee zinnen dezelfde woorden in ongeveer dezelfde verhoudingen gebruiken, niet dat de zinnen hetzelfde betekenen. *the*, *was* en *on* tellen even zwaar als *great* of *rude*, en *not* is gewoon nog een woord."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Knip elke trainingszin in woorden, in kleine letters.
    2. Maak de vocabulary: elk woord dat voorkomt, op alfabet, met de kolom: 0, 1, 2, enzovoort.
    3. Maak van elke zin een vector: tel voor elk woord van de zin 1 op bij de kolom van dat woord. Bewaar alleen de aantallen die geen 0 zijn, als sparse rij.
    4. Maak van een nieuwe zin op dezelfde manier een sparse rij; een woord buiten de vocabulary valt weg.
    5. Meet de cosine similarity van de nieuwe rij met elke trainingsrij, en geef de nieuwe zin het label van de trainingszin met de hoogste. Is elke cosine similarity 0, geef dan geen antwoord.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De code hieronder maakt de vocabulary van drie korte documenten, elk al in woorden geknipt: elk woord komt één keer in de lijst, de lijst wordt gesorteerd, en elk woord krijgt de plaats in de gesorteerde lijst als kolom. De laatste regel laat zien hoe de course een zin in woorden knipt, met `words`.
    """)
    return


@app.cell
def example_cell():
    docs = [["the", "food", "was", "good"], ["the", "service", "was", "slow"], ["good", "food", "good", "service"]]
    vocabulary = []
    for doc in docs:
        for word in doc:
            if word not in vocabulary:
                vocabulary.append(word)
    vocabulary = sorted(vocabulary)
    example_columns = {}
    for i in range(len(vocabulary)):
        example_columns[vocabulary[i]] = i
    print(vocabulary)
    print(example_columns)
    print(C.words("The food was GOOD!"))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De vocabulary heeft zes woorden, van *food* in kolom 0 tot *was* in kolom 5. Het derde document, *good food good service*, wordt de vector `[1, 2, 1, 0, 0, 0]`: *food* één keer, *good* twee keer, *service* één keer, en geen *slow*, *the* of *was*. Als sparse rij is dat `{0: 1, 1: 2, 2: 1}`. `words` geeft de woorden van een tekst in kleine letters en laat alles weg wat geen letter is, zoals *!*; een apostrof binnen een woord blijft staan, zoals in *don't*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_vectorize():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. vectorize: een zin als vector

        `small`, `SMALL` van de course, is de vocabulary van de vier zinnen in de afbeelding, twaalf woorden, en `columns` de vocabulary van de les, 4.518 woorden, uit `columns` van de course; het zijn allebei dictionary's van woord naar kolom. Schrijf `vectorize(words, columns)`: geef een lijst terug met één getal voor elke kolom van `columns`, eerst 0, en verhoog voor elk woord van `words` dat in `columns` staat het getal op plaats `columns[word]` met 1. Een woord dat niet in `columns` staat, valt weg.

        Voor `vectorize(["the", "mic", "is", "great"], small)` is de uitkomst `[0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]`: *great* is kolom 2, *is* 3, *mic* 4 en *the* 9. Vervang de `...`. De check roept dat aan en `vectorize(["crust", "is", "not", "good"], small)`, en daarna eigen gevallen: een woord dat twee keer voorkomt, een woord buiten de vocabulary, helemaal geen woorden, alleen woorden erbuiten, een woord drie keer in een langere zin, een vocabulary van één woord, en een zin over de hele vocabulary van de les.
        """),
        L.task("Maak van een lijst woorden een lijst aantallen, één voor elk woord van een vocabulary."),
    ])
    return


@app.cell
def columns_cell():
    small = C.SMALL
    columns = C.columns()
    print(small)
    print(len(columns), columns["good"], columns["the"])
    print(C.words("The mic is great."))
    return


@app.cell(hide_code=True)
def exercise_vectorize():
    box_vectorize = L.code_box("vectorize", lines=6)
    box_vectorize
    return (box_vectorize,)


@app.cell(hide_code=True)
def check_vectorize(box_vectorize):
    result_vectorize = L.check("vectorize", box_vectorize.value)
    result_vectorize
    return (result_vectorize,)


@app.cell(hide_code=True)
def hints_vectorize():
    L.hints(
        "vectorize",
        [
            "Begin met `vector = [0 for column in columns]`: een 0 voor elke key van `columns`. Loop met `for word in words:` langs de woorden, vraag met `if word in columns:` of het woord een kolom heeft, en tel 1 op bij het getal op die kolom. Geef na de loop `vector` terug.",
            "De kolom van *mic* is `small[\"mic\"]`, en dat is 4, dus het aantal van *mic* is het getal op plaats 4 van `vector`. Hoe schrijf je die plaats voor elk woord?",
            r"`            vector[columns[word]] = ___ + 1`",
        ],
        why=r"`vector = [0 for column in columns]`; `for word in words:` met daarin `if word in columns:` met `vector[columns[word]] = vector[columns[word]] + 1`; na de loop `return vector`.",
    )
    return


@app.cell(hide_code=True)
def text_after_vectorize():
    mo.md(r"""
    In de afbeelding maakte je functie van de vier zinnen lijsten van 4.518 getallen, en van elke lijst zijn er maar twee tot vier geen 0. De kaart toont je vectoren van de eerste 60 trainingszinnen van veraf: een stip voor elk getal dat geen 0 is, op rijen die bijna leeg zijn. Je functie maakte 4.518 getallen per zin om hoogstens vier dingen te zeggen. Een sparse rij zegt hetzelfde met vier. Hier staan de vier zinnen uit de afbeelding als sparse rijen over `small`, `SMALL_ROWS` van de course, en *The mic is great.* als sparse rij over de vocabulary van de les, gemaakt met `sparse_row` van de course. `norm` geeft de lengte van een sparse rij: de wortel uit de som van de kwadraten van de aantallen.
    """)
    return


@app.cell
def rows_cell():
    small_rows = C.SMALL_ROWS
    norm = C.norm
    print(small_rows)
    print(C.sparse_row(C.words("The mic is great.")))
    print(norm(small_rows[0]), norm(small_rows[2]), norm({}))
    return


@app.cell(hide_code=True)
def text_exercise_cosine():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. sparse_cosine: hoeveel lijken twee rijen op elkaar?

        Schrijf `sparse_cosine(a, b)`, met `a` en `b` sparse rijen. Vermenigvuldig voor elke kolom van `a` die ook in `b` staat de twee aantallen, en tel die producten op: dat is het inproduct, want elk ander product is 0. Is de som 0, geef dan `0.0` terug; deel de som anders door `norm(a) * norm(b)`.

        Voor `sparse_cosine({2: 1, 3: 1, 4: 1, 9: 1}, {0: 1, 1: 1, 3: 1, 5: 1})`, de rijen van *The mic is great.* en *Crust is not good.*, is de uitkomst 0,25: de rijen delen alleen kolom 3, *is*, wat 1 × 1 oplevert, en beide lengtes zijn 2. Vervang de `...`. De check roept dat aan en `sparse_cosine({7: 1, 10: 2}, {6: 1, 8: 1, 10: 1, 11: 1})`, en daarna eigen gevallen: een rij met zichzelf, een rij tegen dezelfde rij keer drie, rijen zonder iets gemeen, een lege rij, twee lege rijen, aantallen boven 1 in beide rijen, en twee zinnen over de vocabulary van de les.
        """),
        L.task("Meet de cosine similarity van twee sparse rijen, met alleen de gedeelde kolommen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cosine():
    box_cosine = L.code_box("sparse-cosine", lines=8)
    box_cosine
    return (box_cosine,)


@app.cell(hide_code=True)
def check_cosine(box_cosine):
    result_cosine = L.check("sparse-cosine", box_cosine.value)
    result_cosine
    return (result_cosine,)


@app.cell(hide_code=True)
def hints_cosine():
    L.hints(
        "sparse-cosine",
        [
            "Begin met `dot = 0` en loop met `for column in a:` langs de kolommen. Als `column in b`, tel dan het product van de twee aantallen op bij `dot`. Geef na de loop `0.0` terug als `dot` 0 is, en anders `dot` gedeeld door `norm(a) * norm(b)`.",
            "In kolom 10, *very*, heeft de eerste rij 2 en de tweede 1. Hoeveel telt die kolom op bij het inproduct, en hoe schrijf je dat voor elke kolom?",
            r"`            dot = dot + ___`",
        ],
        why=r"`dot = 0`; `for column in a:` met daarin `if column in b:` met `dot = dot + a[column] * b[column]`; na de loop `if dot == 0:` met `return 0.0`, en dan `return dot / (norm(a) * norm(b))`.",
    )
    return


@app.cell(hide_code=True)
def text_after_cosine():
    mo.md(r"""
    Voor het tweede voorbeeld, *Very, very sad.* en *Service was very prompt.*, geeft je functie 0,447: de zinnen delen alleen *very*, wat 2 × 1 oplevert, en de lengtes zijn √5 en 2. In de afbeelding vergeleek je functie de achtergehouden zin *The delivery was on time.* (*De levering was op tijd.*) met alle 2.400 trainingszinnen. De meest gelijkende is *The bus boy on the other hand was so rude.*, met 0,516: die zin deelt *the*, *was* en *on*, en geen enkel woord over levering of tijd. *The shipping time was also very quick!*, die over hetzelfde gaat, komt op de derde plaats, met 0,507. De meest gelijkende zin is negatief, dus de nieuwe zin zou het verkeerde label krijgen. Korte, gewone woorden zoals *the* en *was* staan in zo veel zinnen dat deze woorden een groot deel van de cosine similarity bepalen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_similar():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. most_similar: de dichtstbijzijnde trainingszin

        `rows` bevat de sparse rijen van de 2.400 trainingszinnen, uit `training_rows` van de course, en `labels` de labels, uit `training_labels`, in dezelfde volgorde: 1 voor positief, 0 voor negatief. Schrijf `most_similar(row, rows)`: meet `cosine(row, rows[i])` voor elke plaats `i` van `rows`, en geef de plaats met de hoogste cosine similarity terug; de eerste zulke plaats als er meer gelijk zijn, en None als geen enkele cosine similarity boven 0 ligt. De check geeft je functie `cosine` van de course, die hetzelfde doet als je `sparse_cosine`.

        Voor `most_similar({8: 1, 10: 1, 3: 1, 1: 1}, small_rows)`, de rij van de woorden *service*, *very*, *is* en *good*, is de uitkomst 1: de cosine similarity met *Crust is not good.* is 0,5, net als met *Service was very prompt.*, en *Crust is not good.* komt eerst. Vervang de `...`. De check roept dat aan en `most_similar({7: 1}, small_rows)`, de rij van *sad*, en daarna eigen gevallen: een rij die met geen enkele rij iets gemeen heeft, twee rijen met dezelfde cosine similarity, de beste rij als laatste, een lege lijst rijen, een lege rij, en een achtergehouden zin tegen de 2.400 trainingsrijen.
        """),
        L.task("Vind de plaats van de rij met de hoogste cosine similarity, of None als geen rij een woord deelt."),
    ])
    return


@app.cell
def training_cell():
    cosine = C.cosine
    rows = C.training_rows()
    labels = C.training_labels()
    print(len(rows), labels[:10])
    print(rows[0])
    return


@app.cell(hide_code=True)
def text_after_training():
    mo.md(r"""
    De eerste trainingszin, *So there is no way for me to plug it in here in the US unless I go by a converter.*, is negatief: zonder verloopstekker kan de schrijver het apparaat in de VS niet aansluiten. De sparse rij heeft 20 kolommen, kolom 1.991 twee keer: *in*. De vector zou daarnaast 4.498 nullen hebben.
    """)
    return


@app.cell(hide_code=True)
def exercise_similar():
    box_similar = L.code_box("most-similar", lines=9)
    box_similar
    return (box_similar,)


@app.cell(hide_code=True)
def check_similar(box_similar):
    result_similar = L.check("most-similar", box_similar.value)
    result_similar
    return (result_similar,)


@app.cell(hide_code=True)
def hints_similar():
    L.hints(
        "most-similar",
        [
            "Begin met `best = None` en met `best_value` op een waarde die alleen een rij met een gedeeld woord kan verslaan, en loop met `for i in range(len(rows)):` langs de plaatsen. Meet `value = cosine(row, rows[i])`; als `value` hoger is dan `best_value`, zet dan `i` in `best` en `value` in `best_value`. Geef na de loop `best` terug.",
            "Begin in gedachten met `best_value = -1`, en neem een rij die geen woord met `row` deelt: de cosine similarity is 0. Laat `value > best_value` die rij door? En hoort dat?",
            r"`    best_value = ___`",
        ],
        why=r"`best = None` en `best_value = 0`; `for i in range(len(rows)):` met `value = cosine(row, rows[i])`, en `if value > best_value:` met `best = i` en `best_value = value`; na de loop `return best`.",
    )
    return


@app.cell(hide_code=True)
def text_after_similar():
    mo.md(r"""
    `labels[most_similar(row, rows)]` is dan het label voor een nieuwe zin. In de afbeelding zocht je functie de meest gelijkende trainingszin bij twaalf achtergehouden zinnen, en het label daarvan klopt bij acht. *The kids are annoying.* vindt *The kids are very cool too.*, en *Setup went very smoothly.* vindt *Very, very sad.*: de woorden die de zinnen delen, zijn niet de woorden die de toon bepalen.

    ### Hoe goed is het?

    Op alle 600 achtergehouden zinnen klopt het label van de meest gelijkende trainingszin bij 397: 66,2%. Een munt zou ongeveer de helft goed hebben. Laat de stop words weg, zoals *the* en *was*, met de lijst uit [Normaliseren: hoofdletters, accenten en stop words](/course/open/nlp-normalize?lang=nl), en er komen er 418 goed uit: 69,7%. Naive Bayes, met dezelfde verdeling en dezelfde aantallen, heeft er 497 goed: 82,8%. Naive Bayes leert van alle 2.400 zinnen samen hoeveel elk woord over het label zegt, terwijl bij de meest gelijkende zin één enkele zin beslist.

    Vectoren met aantallen kunnen naar elke methode die lijsten getallen aanneemt, maar kale aantallen zijn een zwak begin. Manning en collega's raden af om aantallen zonder weging en zonder normalisatie te gebruiken voor classificatie met vectoren. De cosine similarity zorgt voor de lengte; een weging van de woorden ontbreekt nog. Jurafsky en Martin geven aantallen uit vier toneelstukken van Shakespeare: *good* komt 114 keer voor in *As You Like It*, 80 keer in *Twelfth Night*, 62 keer in *Julius Caesar* en 89 keer in *Henry V*, in elk stuk vaker dan *battle*, *fool* of *wit*. Ze wijzen erop dat de twee komedies meer op elkaar lijken dan op de andere twee stukken, met meer *fool* en *wit* dan *battle*. De course berekende de cosine similarity uit die vier aantallen, en kale aantallen verbergen dat verschil: *As You Like It* krijgt 0,950 met de komedie *Twelfth Night*, maar ook 0,949 met *Henry V* en 0,945 met *Julius Caesar*, en dat zijn geen komedies, want *good* beslist bijna alles. Daarna schrijven ze dat information retrieval zulke kale aantallen niet gebruikt.
    """)
    return


@app.cell(hide_code=True)
def compare_picture():
    mo.Html(C.compare_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: je eigen zin

    Typ een zin in het Engels. De afbeelding toont de sparse rij over de vocabulary van de les, de woorden zonder kolom, de vijf meest gelijkende trainingszinnen, en het label van de eerste. Probeer een zin met *not*, één woord, of een Nederlandse zin.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_text = mo.ui.text_area(value=C.DEMO_TEXT, label="Je zin", full_width=True)
    demo_text
    return (demo_text,)


@app.cell(hide_code=True)
def demo_look(demo_text):
    mo.Html(C.demo_picture(demo_text.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Bij de zin in het vak delen de bovenste twee de eerste plaats: *The food was very good.* en *The service was extremely slow.* hebben allebei een cosine similarity van 0,744 met de zin, de ene positief en de andere negatief. De eerste in de lijst wint, dus het antwoord is *positief*, door de volgorde van de zinnen in het bestand. Een zin die lof en klachten mengt, heeft geen enkele meest gelijkende zin die er echt bij past.
    """)
    return


@app.cell(hide_code=True)
def text_explore():
    mo.md(r"""
    ### Probeer het: andere vocabulary's

    Deze les maakte de vocabulary van elk woord in kleine letters, en telde hoe vaak elk woord voorkomt. Verander die keuzes, en kijk hoeveel kolommen de vocabulary krijgt, hoe sparse de rijen zijn, en hoeveel van de 600 achtergehouden zinnen het goede label krijgen.
    """)
    return


@app.cell(hide_code=True)
def explore_controls():
    explore_lower = mo.ui.checkbox(value=True, label="Elk woord in kleine letters")
    explore_stop = mo.ui.checkbox(value=False, label="Stop words weglaten")
    explore_binary = mo.ui.checkbox(value=False, label="0 of 1 in plaats van een aantal")
    explore_rare = mo.ui.checkbox(value=False, label="Woorden die maar één keer voorkomen weglaten")
    mo.vstack([explore_lower, explore_stop, explore_binary, explore_rare])
    return explore_binary, explore_lower, explore_rare, explore_stop


@app.cell(hide_code=True)
def explore_look(explore_binary, explore_lower, explore_rare, explore_stop):
    mo.Html(C.explore_picture(explore_lower.value, explore_stop.value, explore_binary.value, explore_rare.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_explore():
    mo.md(r"""
    Zonder kleine letters krijgen *Great* en *great* twee kolommen: 5.149 kolommen, en 64,7% goed. Stop words weglaten helpt het meest, 69,7%, met 4.364 kolommen. Woorden die maar één keer voorkomen weglaten, brengt de vocabulary terug tot 1.893 kolommen, en de score blijft 66,2%: een woord dat in één trainingszin staat, kan alleen bij die ene zin passen. Geen van deze keuzes, alleen of samen, verschuift de score met meer dan 21 zinnen; het gat met naive Bayes blijft.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Oefening 4. Wat komt eruit?

    Elke regel is een paar teksten. Zet het paar bij de cosine similarity van de twee rijen, over de vocabulary van deze les.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_comes_out = L.sort("what-comes-out")
    sort_comes_out
    return (sort_comes_out,)


@app.cell(hide_code=True)
def check_sort(sort_comes_out):
    result_sort = L.check("what-comes-out", sort_comes_out)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Spam of niet: naive Bayes](/course/open/lm-naive-bayes?lang=nl) hield voor spam en voor de andere berichten elk een bag of words bij; deze les schreef de bag of words van elke zin als een vector over een vaste vocabulary, bewaarde die vector als sparse rij, en vergeleek twee zulke rijen met de cosine similarity uit [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl). Kleine letters en een lijst stop words, zoals in [Normaliseren: hoofdletters, accenten en stop words](/course/open/nlp-normalize?lang=nl), veranderen welke kolommen er zijn. Kale aantallen laten gewone woorden zoals *the* te veel beslissen. De volgende les van het domein, *TF-IDF: woorden wegen*, weegt elk woord naar het aantal documenten waarin het woord voorkomt.

    **Een voorbeeld uit het dagelijks leven.** Een nieuwssite die onder een bericht *verwante berichten* laat zien, kan zulke berichten zo vinden: maak van elk bericht een vector met aantallen woorden, en toon de berichten met de hoogste cosine similarity.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_cosine, result_similar, result_sort, result_vectorize):
    L.wrap_up(result_vectorize, result_cosine, result_similar, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
